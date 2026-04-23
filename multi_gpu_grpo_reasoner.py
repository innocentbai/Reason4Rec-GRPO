#!/usr/bin/env python3
"""
Multi-GPU GRPO Training Script for Reasoner Model
基于TRL框架对微调好的Reasoner进行4-GPU强化学习训练
"""

import os
import pandas as pd
import argparse
import torch
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from peft import PeftModel
from accelerate import PartialState
from utils import logits_weighted_predict

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 减少警告输出
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def prepare_grpo_dataset(dataset_path, history_path, product_class):
    """准备GRPO训练数据集"""
    # 加载数据
    data_df = pd.read_pickle(dataset_path)
    history_df = pd.read_pickle(history_path)
    
    
    prompts = []
    
    for idx, row in data_df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        target_title = row['title']
        
        # 获取用户历史
        user_history = history_df[history_df['user_id'] == user_id]
        user_history = user_history[user_history['unixReviewTime'] < row['unixReviewTime']]
        user_history = user_history.sort_values(by='unixReviewTime', ascending=True)
        user_history = user_history.tail(10)

        # 获取物品历史
        item_history = history_df[history_df['item_id'] == item_id]
        item_history = item_history[item_history['unixReviewTime'] < row['unixReviewTime']]
        item_history = item_history.sort_values(by='unixReviewTime', ascending=True)
        item_history = item_history.tail(10)

        # 构建用户历史文本
        user_history_text = ''
        for i, (_, his) in enumerate(user_history.iterrows()):
            his_title = his['title']
            user_history_text += f"{i + 1}. {his_title}\n"
            user_history_text += f"{his['aspect_preference_summary']}".strip() + '\n\n'
        user_history_text = user_history_text.strip()
        
        # 构建物品历史文本
        item_history_text = ''
        for i, (_, his) in enumerate(item_history.iterrows()):
            his_title = his['title']
            item_history_text += f"{i + 1}. {his_title}\n"
            item_history_text += f"{his['aspect_preference_summary']}".strip() + '\n\n'
        item_history_text = item_history_text.strip()

        # 构建提示词
        prompt = f"""Here is information about a user and a new {product_class} "{target_title}" being recommended to the user. For the user, we have the user's review history. For the new item being recommended, we have the item review history by other users.

### User Review History ###
{user_history_text}

### Item Review History by other users ###
{item_history_text}

Analyze whether the user will like the new {product_class} "{target_title}" based on the user's preferences and the recommended item's features. Give you rationale in one paragraph."""

        prompts.append({
            "prompt": prompt,
            "user_id": user_id,
            "item_id": item_id,
            "target_title": target_title,
            "target_rating": float(row['ratings']),
            "user_average_rating": float(user_history['ratings'].mean()) if len(user_history) > 0 else 4.0,
            "item_average_rating": float(item_history['ratings'].mean()) if len(item_history) > 0 else 4.0
        })
    
    return Dataset.from_list(prompts)

def load_model_and_tokenizer_distributed(base_model_path, adapter_path=None, load_in_4bit=True):
    """
    在分布式环境中加载模型和tokenizer
    """
    # 获取当前进程的device
    device_index = PartialState().process_index
    
    # 从适配器路径加载tokenizer（如果存在）
    if adapter_path and os.path.exists(os.path.join(adapter_path, 'tokenizer.json')):
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 设置量化配置 (对于已量化模型直接跳过)
    if 'bnb-4bit' in base_model_path:
        if PartialState().is_main_process:
            print("Model already quantized, loading without additional quantization_config")
        quantization_config = None
    elif load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        quantization_config = None
    
    # 加载基础模型 - 关键：使用正确的device_map
    if PartialState().is_main_process:
        print(f"Loading base model from: {base_model_path} on device {device_index}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map={'': device_index},  # 关键：确保模型在正确的GPU上
            trust_remote_code=True
        )
    except Exception as e:
        if PartialState().is_main_process:
            print(f"Failed to load with quantization_config: {e}")
            print("Retrying without quantization_config...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map={'': device_index},
            trust_remote_code=True
        )
    
    # 加载PEFT适配器（如果指定）
    if adapter_path:
        if PartialState().is_main_process:
            print(f"Loading PEFT adapters from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer

def create_reward_function_distributed(base_model_path, predictor_adapter_path, product_class):
    """创建分布式奖励函数"""
    
    # 加载预测器模型 - 在每个GPU上都加载
    predictor, tokenizer = load_model_and_tokenizer_distributed(
        base_model_path, 
        predictor_adapter_path,
        load_in_4bit=True
    )
    predictor.eval()
    
    # 预加载历史数据以避免重复读取
    history_df = pd.read_pickle('./Data/Music_data/train_summarizer_generation_results.pkl')
    
    def reward_function(completions, **kwargs):
        """
        奖励函数：基于预测器模型计算奖励
        """
        rewards = []
        
        # 从kwargs中获取必要信息
        user_ids = kwargs.get('user_id', [])
        item_ids = kwargs.get('item_id', [])
        target_titles = kwargs.get('target_title', [])
        target_ratings = kwargs.get('target_rating', [])
        user_avg_ratings = kwargs.get('user_average_rating', [])
        item_avg_ratings = kwargs.get('item_average_rating', [])
        
        for i, completion in enumerate(completions):
            try:
                # 构建预测器的输入
                user_id = user_ids[i] if i < len(user_ids) else None
                item_id = item_ids[i] if i < len(item_ids) else None
                target_title = target_titles[i] if i < len(target_titles) else ""
                target_rating = target_ratings[i] if i < len(target_ratings) else 3.0
                user_avg_rating = user_avg_ratings[i] if i < len(user_avg_ratings) else 3.0
                item_avg_rating = item_avg_ratings[i] if i < len(item_avg_ratings) else 3.0
                
                if user_id is None or item_id is None:
                    rewards.append(0.0)
                    continue
                
                # 重新构建历史信息（带评分）
                user_history = history_df[history_df['user_id'] == user_id].tail(10)
                item_history = history_df[history_df['item_id'] == item_id].tail(10)
                
                user_history_text = ''
                for j, (_, his) in enumerate(user_history.iterrows()):
                    his_title = his['title']
                    his_rating = his['ratings']
                    user_history_text += f"{j + 1}. {his_title}, {float(his_rating):.1f};\n"
                    user_history_text += f"{his['aspect_preference_summary']}".strip() + '\n\n'
                user_history_text = user_history_text.strip()
                
                item_history_text = ''
                for j, (_, his) in enumerate(item_history.iterrows()):
                    his_title = his['title']
                    his_rating = his['ratings']
                    item_history_text += f"{j + 1}. {his_title}, {float(his_rating):.1f};\n"
                    item_history_text += f"{his['aspect_preference_summary']}".strip() + '\n\n'
                item_history_text = item_history_text.strip()
                
                # 构建预测器提示词
                predictor_prompt = f"""Here is information about a user and a new {product_class} "{target_title}" being recommended to the user. For the user, we have the user's past rating history User ratings range from 1 to 5, where 1 is the lowest and 5 is the highest. For the new item being recommended, we have the item rating history by other users.

### User Rating History ###
{user_history_text}

### Item Rating History by other users ###
{item_history_text}

### Average Past Ratings ###
User's Average Rating (all previous ratings): {user_avg_rating:.1f}
Item's Average Rating (all ratings by other users): {item_avg_rating:.1f}

### Personalized Recommendation Analysis ###
{completion}

Based on the above information, please predict the user's rating for "{target_title}", (1 being the lowest and 5 being highest, directly give the rating without other content.)
[Output Format] Predicted Rating: [A rating between 1 and 5]\tGive your reply following the output format without any extra information."""
                
                # 使用预测器预测评分
                output_prefix = "Predicted Rating: "
                predicted_rating = logits_weighted_predict(predictor, tokenizer, predictor_prompt, output_prefix)
                
                # 计算奖励：预测准确度越高，奖励越大
                reward = -abs(predicted_rating - target_rating)  # 负的绝对误差
                rewards.append(reward)
                
            except Exception as e:
                # 只在主进程打印错误以避免重复日志
                if PartialState().is_main_process:
                    print(f"Error in reward calculation for completion {i}: {e}")
                rewards.append(0.0)
        
        return rewards
    
    return reward_function

def main():
    parser = argparse.ArgumentParser(description="4-GPU GRPO Training for Reasoner")
    parser.add_argument("--dataset", type=str, default="Music_data", help="Dataset name")
    parser.add_argument("--base_model_path", type=str,
                       default="/data/wdh/unsloth/llama-3-8b-Instruct-bnb-4bit",
                       help="Path to the base model")
    parser.add_argument("--reasoner_adapter_path", type=str, 
                       default="./checkpoints/Music_data/Reasoner/final_checkpoint",
                       help="Path to the reasoner adapter")
    parser.add_argument("--predictor_adapter_path", type=str,
                       default="./checkpoints/Music_data/Predictor/final_checkpoint", 
                       help="Path to the predictor adapter")
    parser.add_argument("--output_dir", type=str, 
                       default="./checkpoints/Music_data/Reasoner_4GPU_GRPO",
                       help="Output directory for GRPO trained model")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.01, help="KL coefficient")
    parser.add_argument("--num_generations", type=int, default=4, help="Number of generations per prompt")
    parser.add_argument("--max_completion_length", type=int, default=256, help="Max completion length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--save_steps", type=int, default=50, help="Save steps")
    parser.add_argument("--logging_steps", type=int, default=5, help="Logging steps")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Load model in 4bit")
    
    args = parser.parse_args()
    
    # 设置优化环境变量
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    os.environ["NCCL_NET_GDR_LEVEL"] = "PHB"
    os.environ["NCCL_SOCKET_NTHREADS"] = "4"
    
    # 启用优化
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 数据集路径
    dataset_path = f'./Data/{args.dataset}/distilling_high_quality_reasons.pkl'
    history_path = f'./Data/{args.dataset}/train_summarizer_generation_results.pkl'
    product_class = 'Digital Music' if args.dataset == 'Music_data' else 'Product'
    
    # 只在主进程打印配置信息
    if PartialState().is_main_process:
        print("=== 4-GPU GRPO Training Configuration ===")
        print(f"Dataset: {args.dataset}")
        print(f"Base Model: {args.base_model_path}")
        print(f"Reasoner Adapter: {args.reasoner_adapter_path}")
        print(f"Predictor Adapter: {args.predictor_adapter_path}")
        print(f"Output Directory: {args.output_dir}")
        print(f"Per Device Batch Size: {args.per_device_train_batch_size}")
        print(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
        print(f"Effective Batch Size: {args.per_device_train_batch_size * 4 * args.gradient_accumulation_steps}")
        print(f"Learning Rate: {args.learning_rate}")
        print(f"Beta (KL coefficient): {args.beta}")
        print(f"Num Generations: {args.num_generations}")
        print(f"Max Completion Length: {args.max_completion_length}")
        print(f"Temperature: {args.temperature}")
        print(f"Load in 4bit: {args.load_in_4bit}")
        print("=========================================")
    
    # 准备数据集
    if PartialState().is_main_process:
        print("Preparing dataset...")
    train_dataset = prepare_grpo_dataset(
        dataset_path,
        history_path, 
        product_class
    )
    if PartialState().is_main_process:
        print(f"Dataset size: {len(train_dataset)}")
    
    # 创建奖励函数
    if PartialState().is_main_process:
        print("Creating reward function...")
    reward_func = create_reward_function_distributed(
        args.base_model_path, args.predictor_adapter_path, product_class
    )
    
    # GRPO配置 - 针对4-GPU优化
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        
        # GRPO特定参数
        beta=args.beta,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        epsilon=0.2,
        loss_type="bnpo",
        scale_rewards=True,
        mask_truncated_completions=True,
        
        # 生成参数
        top_p=0.95,
        repetition_penalty=1.0,
        
        # 分布式训练优化
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        
        # 优化器配置
        optim="adamw_torch",
        warmup_ratio=0.1,
        
        # 日志参数
        log_completions=True if PartialState().is_main_process else False,
        num_completions_to_print=2,
        report_to=["wandb"],
        logging_first_step=True,
        save_only_model=True,
        
        # 4-GPU特定优化
        ddp_find_unused_parameters=False,
    )
    
    # 加载 reasoner 模型
    if PartialState().is_main_process:
        print("Loading reasoner model...")
    reasoner_model, reasoner_tokenizer = load_model_and_tokenizer_distributed(
        args.base_model_path,
        args.reasoner_adapter_path,
        load_in_4bit=args.load_in_4bit
    )
    
    # 创建trainer
    trainer = GRPOTrainer(
        model=reasoner_model,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # 开始训练
    if PartialState().is_main_process:
        print("Starting 4-GPU GRPO training...")
    trainer.train()
    
    # 保存最终模型 (只在主进程保存)
    if PartialState().is_main_process:
        print("Saving final model...")
        trainer.save_model()
        print("4-GPU GRPO training completed!")

if __name__ == "__main__":
    main()