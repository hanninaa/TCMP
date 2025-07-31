import torch
import numpy as np
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import json
from torch.utils.data import DataLoader
from typing import Dict, Any
from openai import OpenAI

# 定义 system_prompt 和 user_prompt
system_prompt = """You are an AI assistant with extensive knowledge in the field of medicine. I want you act as an answer judge.
Given a question and an answer, your objective is to determine if the provided answer contains non-factual or hallucinated information. 
You SHOULD give your judgement based on the following the world knowledge.
You are trying to determine if there is a factual contradiction between the answer and the world knowledge. Some information in the answer might be fabricated.
Answer '0' if the answer is factual and '1' if the answer is hallucinated. Please return the result in JSON format. The output must be a JSON object. Include ‘result'. """

user_prompt = """ 
World Knowledge:{knowledge}
Question:{question}
Answer:{answer} Return '0' if the answer is factual and '1' if the answer is hallucinated. No explanation is needed."""


# 调用API来进行推理
client = OpenAI(api_key="sxxx", base_url="xxx")



with open(r'xxx') as f:
    train_data = json.load(f)
results = []

for item in train_data:
    question = item.get('questionAsk', '')
    answer = item.get('hallucination_generation', '')
    context = item.get('context', '')
    search_results = item.get('search_results', '')
    formatted_user_prompt = user_prompt.format(knowledge=search_results, question=question, answer=answer)


    response = client.chat.completions.create(
        model="qwen3-32b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_user_prompt}
        ],
        temperature=0.2,
        max_tokens=100,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content
    print(content)

    try:
        result = json.loads(content)
        results.append(result.get('result', None))
    except Exception as e:
        results.append(None)


# 计算1的占比
num_1 = sum(1 for r in results if str(r) == '1')
num_valid = sum(1 for r in results if r is not None)
ratio = num_1 / num_valid if num_valid > 0 else 0
print(f"1的占比: {ratio:.2f}")