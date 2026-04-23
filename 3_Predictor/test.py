import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pandas as pd
from unsloth import FastLanguageModel
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import chat_with_LLM_manyreply
from utils import chat_with_LLM
from utils import logits_weighted_predict

dataset = 'Music_data'
product_class = 'Digital Music'
data_df = pd.read_pickle(f'./Data/{dataset}/raw_data/test.pkl')
history_df = pd.read_pickle(f'./Data/{dataset}/train_summarizer_generation_results.pkl')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./checkpoints/Music_data/Reasoner/final_checkpoint", 
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True
)
FastLanguageModel.for_inference(model)

row = data_df.iloc[992]

user_id = row['user_id']
item_id = row['item_id']
target_title = row['title']
    
user_history = history_df[history_df['user_id'] == user_id]
user_history = user_history[user_history['unixReviewTime'] < row['unixReviewTime']]
user_history = user_history.sort_values(by='unixReviewTime', ascending=True)
user_history = user_history.tail(10)

item_history = history_df[history_df['item_id'] == item_id]
item_history = item_history[item_history['unixReviewTime'] < row['unixReviewTime']]
item_history = item_history.sort_values(by='unixReviewTime', ascending=True)
item_history = item_history.tail(10)

user_history_text = ''
for i, (_, his) in enumerate(user_history.iterrows()):
    his_title = his['title']
    user_history_text += f"{i + 1}. {his_title}\n"
    user_history_text += f"{his['aspect_preference_summary']}".strip() + '\n\n'
user_history_text = user_history_text.strip()
    
item_history_text = ''
for i, (_, his) in enumerate(item_history.iterrows()):
    his_title = his['title']
    item_history_text += f"{i + 1}. {his_title}\n"
    item_history_text += f"{his['aspect_preference_summary']}".strip() + '\n\n'
item_history_text = item_history_text.strip()

question = f"""Here is information about a user and a new {product_class} "{target_title}" being recommended to the user. For the user, we have the user's review history. For the new item being recommended, we have the item review history by other users.

### User Review History ###
{user_history_text}

### Item Review History by other users ###
{item_history_text}

Analyze whether the user will like the new {product_class} "{target_title}" based on the user's preferences and the recommended item's features. Give you rationale in one paragraph."""

#单个回答
reply=chat_with_LLM(model, tokenizer, question, temperature=0)
# Reasoner 生成多份回答
reasoner_replies = chat_with_LLM_manyreply(model, tokenizer, question, temperature=0.8, num_return_sequences=20)


# 加载 predictor 模型
predictor, pred_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./checkpoints/Music_data/Predictor/final_checkpoint", 
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True
)
FastLanguageModel.for_inference(predictor)


output_prefix = "Predicted Rating: "
single_prompt = f"""Here is information about a user and a new {product_class} "{target_title}" being recommended to the user. For the user, we have the user's past rating history. For the new item being recommended, we have the item rating history by other users.

### User Rating History ###
{user_history_text}

### Item Rating History by other users ###
{item_history_text}

### Personalized Recommendation Analysis ###
{reply}

Based on the above information, please predict the user's rating for "{target_title}", (1 being the lowest and 5 being highest, directly give the rating without other content.)
[Output Format] Predicted Rating: [A rating between 1 and 5]\tGive your reply following the output format without any extra information."""

score = logits_weighted_predict(predictor, pred_tokenizer, single_prompt, output_prefix)

print("用户真实评分:", row['ratings']) 
print(f"====temperature==0时,生成唯一reply====")
print(reply)
print(f"Predicted Rating: {score}")


# 预测每个回复的得分

scores = []
for i, reasoner_reply in enumerate(reasoner_replies):
    rating_prompt = f"""Here is information about a user and a new {product_class} "{target_title}" being recommended to the user. For the user, we have the user's past rating history. For the new item being recommended, we have the item rating history by other users.

### User Rating History ###
{user_history_text}

### Item Rating History by other users ###
{item_history_text}

### Personalized Recommendation Analysis ###
{reasoner_reply}

Based on the above information, please predict the user's rating for "{target_title}", (1 being the lowest and 5 being highest, directly give the rating without other content.)
[Output Format] Predicted Rating: [A rating between 1 and 5]\tGive your reply following the output format without any extra information."""
    
    score = logits_weighted_predict(predictor, pred_tokenizer, rating_prompt, output_prefix)
    scores.append(score)

# 打印输出
for i in range(20):

    print(f"\n==== Reply {i + 1} ====")
    print(reasoner_replies[i])
    print(f"Predicted Rating: {scores[i]}")



