#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合脚本：读取 TCMP_HALLU.jsonl → 清洗 → 抽取 10k 条 → 调用 DeepSeek API 把填空题改为问句 → 保存
"""

import json
import re
import jsonlines
from openai import OpenAI
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ------------------------------------------------------------------
# 1. 路径与参数
# ------------------------------------------------------------------
SRC_JSONL = r'F:/XXX'
CLEAN_JSON = r'F:/XXX/demo.json'
GPT_IN_JSON  = CLEAN_JSON           # 第二次加载的文件
GPT_OUT_JSON = r'F:/XXX/demo_gpt.json'

API_KEY  = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
BASE_URL = "XXX"

N_SAMPLE = 10_000                 # 抽取多少条做 demo
MAX_WORKERS = 16                  # 并发线程数，可按机器调整

# ------------------------------------------------------------------
# 2. 读取 & 清洗原始数据
# ------------------------------------------------------------------
def clean_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """对单条 original_data 进行字段清洗"""
    original = rec.get("original_data", {})
    # 删除 instruction
    original.pop("instruction", None)

    # 清理 question
    if "question" in original:
        original["question"] = (
            original["question"]
            .split("答案：")[0]
            .split("\nA.")[0]
            .strip()
        )

    # 清理 long_answer
    if "long_answer" in original:
        original["long_answer"] = re.sub(r"^答案：", "", original["long_answer"]).strip()

    # 清理 context
    if "context" in original:
        original["context"] = re.sub(r"^该问题解析：", "", original["context"]).strip()

    return rec


def load_and_clean(path: str) -> list:
    """读取 jsonl 并清洗"""
    with jsonlines.open(path, "r") as reader:
        data = [clean_record(obj) for obj in reader]
    return data


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


print("Step-1 读取并清洗原始数据 ...")
cleaned = load_and_clean(SRC_JSONL)
demo = cleaned[:N_SAMPLE]
save_json(demo, CLEAN_JSON)
print(f"已保存 {len(demo)} 条清洗后数据 -> {CLEAN_JSON}")

# ------------------------------------------------------------------
# 3. 调用 DeepSeek API 把填空题转换为问句
# ------------------------------------------------------------------
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = (
    "请将以下填空题转换为适合中医药领域的问句格式。\n"
    "返回纯 JSON，形如：\n"
    '{"questionAsk": "<转换后的问题>"}'
)

def convert_question(question: str) -> str:
    """调用大模型并返回 questionAsk"""
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            temperature=0.2,
            max_tokens=256,
            response_format={"type": "json_object"}
        )
        result = json.loads(resp.choices[0].message.content)
        return result.get("questionAsk", question)
    except Exception as e:
        print("API error:", e)
        return question      # 兜底返回原问题

# ------------------------------------------------------------------
# 4. 并发处理
# ------------------------------------------------------------------
print("Step-2 加载待处理 demo ...")
with open(GPT_IN_JSON, "r", encoding="utf-8") as f:
    demo_texts = json.load(f)

def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    q = item.get("original_data", {}).get("question", "")
    item["original_data"]["questionAsk"] = convert_question(q)
    return item

print("Step-3 并行调用 API ...")
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
    futures = [exe.submit(process_item, it) for it in demo_texts]
    for _ in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
        pass          # tqdm 仅做进度展示

# ------------------------------------------------------------------
# 5. 保存最终结果
# ------------------------------------------------------------------
save_json(demo_texts, GPT_OUT_JSON)
print(f"全部完成！结果已保存到 -> {GPT_OUT_JSON}")