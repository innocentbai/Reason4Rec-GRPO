import openai
from time import sleep
import torch
import torch.nn.functional as F
import os
openai.api_key = os.getenv("DEEPSEEK_API_KEY", "your_deepseek_key")
openai.base_url = "https://api.deepseek.com"

def chat_with_LLM(model, tokenizer, question, max_new_tokens=1024, temperature=0.2):
    messages = [{
        "role": "user",
        "content": question
    }]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    if temperature != 0:
        generated_ids = model.generate(
            **model_inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True,
            temperature=temperature,
            eos_token_id=[tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            pad_token_id=tokenizer.eos_token_id
        )
    else:
        generated_ids = model.generate(
            **model_inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False,
            top_p=None,
            temperature=None,
            eos_token_id=[tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            pad_token_id=tokenizer.eos_token_id
        )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response



def logits_weighted_predict(model, tokenizer, question, output_prefix):
    messages = [{
        "role": "user",
        "content": question
    }]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += output_prefix

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    next_token_logits  = logits[0, -1, :]

    tokens = ["1", "2", "3", "4", "5"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    predicted_logits = next_token_logits[token_ids]
    normalized_probs = F.softmax(predicted_logits, dim=0)

    ratings = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float).to(normalized_probs.device)
    predicted_rating = torch.sum(normalized_probs * ratings)
    results = predicted_rating.item()
    if results < 1:
        results = 1
    elif results > 5:
        results = 5
    return results

def chat_with_gpt(question, max_tries=1, model = 'deepseek-chat', temperature = 0.2):
    messages = []
    messages.append({"role": "user", "content": question})
    response = None   
    for i in range(max_tries):
        try:
            response = openai.ChatCompletion.create(
                model=model,  
                messages=messages,
                temperature=temperature,  
                n=1, 
                stop=None,  
                timeout=None,  
                max_tokens=4096 
            )
            break
        except Exception as e:
            print(f"Attempt {i+1} failed with error: {e}")
            if i < max_tries:  
                sleep(5)  
            else:
                raise e
    if response is None:
        reply = None
    else:
        reply = response.choices[0].message.content
    return reply



def chat_with_LLM_manyreply(model, tokenizer, question, max_new_tokens=1024, temperature=0.2, num_return_sequences=1):
    messages = [{
        "role": "user",
        "content": question
    }]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    do_sample = temperature > 0
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "num_return_sequences": num_return_sequences,
        "eos_token_id": [tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        "pad_token_id": tokenizer.eos_token_id,
    }

    generated_ids = model.generate(**model_inputs, **generate_kwargs)

    # 去除每条生成中 prompt 部分
    input_ids = model_inputs.input_ids.repeat_interleave(num_return_sequences, dim=0)
    generated_ids = [
        output_ids[len(input_id):] for input_id, output_ids in zip(input_ids, generated_ids)
    ]

    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return responses
