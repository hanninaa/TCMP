import json
import time
from openai import OpenAI

# ========== 用户配置 ==========
INPUT_FILE = r"F:\Nijia.Han23\PycharmProjects\Medhall\MedHallu\Datasets\pqa_tcm\demo_gpt_questionAsk.json"              # 输入文件名
OUTPUT_FILE = r"F:\Nijia.Han23\PycharmProjects\Medhall\MedHallu\Datasets\pqa_tcm\classified_output.json" # 输出文件名
MODEL_NAME = "gpt-4.1"                  # 模型名称（推荐 gpt-4o 或 gpt-4-turbo）
MAX_RETRIES = 3                        # 最大重试次数
SLEEP_BETWEEN_CALLS = 1                # 每次请求间隔（秒）
# =================================

client = OpenAI(api_key="xxx", base_url="xxx")


# ---------- 分类Prompt ----------
def build_prompt(question):
    return f"""
你是一个中医药问题分类助手。
请根据用户输入的问题，将其归入以下三类之一，仅输出数字编号（1、2、3），不需要文字解释。

【分类标准】

1. 药物适应症与功效
——包括药物的适应症、功效、药理作用、治疗作用等问题。
示例：
- 抗病毒片适用于哪些中医疾病？
- 茵陈五苓糖浆适用于哪些西医疾病？
- 柴胡舒肝丸的功效是什么？
- 吴茱萸汤具有哪些治疗作用？

2. 药物成分与配方
——包括药物的成分、主要成分、方剂组成等问题。
示例：
- 藿香正气丸的主要成分有哪些？
- 五苓散片中的君药是什么？
- 三鞭酒含有哪些中药材？

3. 药物分类与机制
——包括药物的分类、作用机制、相关靶点基因等问题。
示例：
- 蛇胆川贝散属于哪一类中医药？
- 藿香正气丸与哪些靶点基因相关？
- 苦甘颗粒的药理机制是什么？

【输出要求】
只返回一个JSON对象，格式如下：
{{
  "category_id": 1 或 2 或 3
}}

【请分类以下问题】：
{question}
"""

# ---------- 读取输入数据 ----------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    questions = json.load(f)

# ---------- 尝试加载已有结果（断点续传） ----------
try:
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        classified_results = json.load(f)
        # 如果之前已经有 summary 字段，去掉
        if isinstance(classified_results, dict) and "summary" in classified_results:
            classified_results = classified_results["data"]
except FileNotFoundError:
    classified_results = []

processed_ids = {item["id"] for item in classified_results if "category_id" in item}

# ---------- 主循环 ----------
for item in questions[:5]:
    qid = item["id"]
    question = item["question"]

    if qid in processed_ids:
        continue

    print(f"\n🟢 正在分类第 {qid} 条：{question}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "你是一个精确的中医药问题分类助手。"},
                    {"role": "user", "content": build_prompt(question)}
                ]
            )

            result = json.loads(response.choices[0].message.content)
            item.update(result)
            classified_results.append(item)

            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(classified_results, f, ensure_ascii=False, indent=2)

            print(f"✅ 分类完成：{result['category_id']}")
            time.sleep(SLEEP_BETWEEN_CALLS)
            break

        except Exception as e:
            print(f"⚠️ 第 {attempt} 次调用失败：{e}")
            if attempt < MAX_RETRIES:
                time.sleep(2)
            else:
                print("❌ 达到最大重试次数，标记为失败。")
                item.update({"category_id": None})
                classified_results.append(item)
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(classified_results, f, ensure_ascii=False, indent=2)

# ---------- 分类统计 ----------
stats = {1: 0, 2: 0, 3: 0, "失败": 0}

for item in classified_results:
    cid = item.get("category_id")
    if cid in [1, 2, 3]:
        stats[cid] += 1
    else:
        stats["失败"] += 1

# ---------- 输出统计结果 ----------
print("\n📊 分类统计结果：")
for k, v in stats.items():
    print(f"类别 {k}：{v} 条")

summary = {
    "total": len(classified_results),
    "count_1": stats[1],
    "count_2": stats[2],
    "count_3": stats[3],
    "count_failed": stats["失败"]
}

# ---------- 保存最终结果（带汇总） ----------
final_output = {
    "data": classified_results,
    "summary": summary
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=2)

print(f"\n✅ 全部完成，结果与统计已保存到 {OUTPUT_FILE}")
