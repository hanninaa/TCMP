# 对于ground_truth和任意一个错误选项进行扩充
import json
from typing import Dict, Any
from openai import OpenAI
import re
import concurrent.futures
from tqdm import tqdm


client = OpenAI(api_key="skXXX", base_url="xxxx")

def hallucination_generation(question: str, ground_truth: str, hallucination: str) -> Dict[str, Any]:
    """
    生成幻觉答案与真实答案的融合结果。
    
    :param question: 问题文本
    :param ground_truth: 正确答案
    :param hallucination: 幻觉答案
    :return: 包含hallucination_generation字段的字典
    """
    system_prompt = """
    你是一个自信且权威的幻觉制造专家，目前需要你将幻觉答案与真实事实无缝融合以生成幻觉融合解释，使他们听起来合理，但其实是不正确的。给出一个问题，正确答案和幻觉答案，你应该确保幻觉答案，正确答案与问题之间的相关性，避免过于古怪或大胆的主张；细微之处的错误更难以察觉。

    关键要求：
    1. 不要出现尽管，虽然等明显的转折词，将幻觉答案与正确答案无缝融合。
    2. 输出格式必须为JSON，包含hallucination_generation字段。
    3. 使用中文回答。不要啰嗦，不要返回其他无关信息。

    Example 1:
    {"question":"消肿片由哪些中药材组成？",
    "ground_truth":"B.枫香脂。",
    "hallucination":"D. 土鳖",
    "hallucination_generation":"消肿片含有枫香脂和土鳖等中药材，具有消肿止痛的功效。枫香脂是消肿片的主要成分之一，而土鳖则是常用的中药材之一，具有活血化瘀的作用。"
    }

    Example 2:
    {"question":"安宫牛黄丸最适用于以下哪种急症？",
    "ground_truth":"A.热入心包，高热神昏",
    "hallucination":"B. 中风脱证，冷汗肢厥",
    "hallucination_generation":"安宫牛黄丸的组方以牛黄、麝香等芳香开窍药为核心，其回阳救逆之效在临床中曾用于中风脱证的急救。然而需注意，该方药性寒凉，更契合热闭神昏的证候，脱证宜选参附汤类温补固脱。"
    }

    Example 3:
    {"question":"在疏风活络丸的配方中，甘草扮演了什么样的角色？",
    "ground_truth":"B. 使药。",
    "hallucination":"C. 佐药",
    "hallucination_generation":"甘草在本方中既作为佐药制约主药的燥烈之性，又作为使药引导诸药归经，但现代方剂学研究更强调其佐制药性的关键作用，因此部分权威配伍理论将其首要角色定位为佐药，而传统分类则归为使药。"
    }
    """

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"问题：{question}\n正确答案：{ground_truth}\n幻觉答案：{hallucination}"}
        ],
        temperature=0.2,
        max_tokens=500,
        response_format={"type": "json_object"}
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return result
    except json.JSONDecodeError:
        return {"hallucination_generation": "Invalid JSON response"}
    
def process_item(item):
    question = item.get('questionAsk', '')
    ground_truth = item.get('long_answer', '')
    hallucination = item.get('hallucination_answer', '')
    result = hallucination_generation(question, ground_truth, hallucination)
    item['hallucination_generation'] = result.get('hallucination_generation', '')
    return item



with open(r'F:\Nijia.Han23\PycharmProjects\Medhall\MedHallu\Datasets\pqa_tcm\demo_bak.json', 'r', encoding='utf-8') as f:
    texts = json.load(f)
    print(f"Loaded {len(texts)} items from the dataset after saving.")

# 使用并行化处理
with concurrent.futures.ThreadPoolExecutor() as executor:
    # 使用 tqdm 包装 executor.map 以显示进度条
    for processed_item in tqdm(executor.map(process_item, texts), total=len(texts), desc="Processing items"):
        # 每处理完一条数据，立即保存
        try:
            with open(r'F:\Nijia.Han23\PycharmProjects\Medhall\MedHallu\Datasets\pqa_tcm\demo_gpt.json', 'w', encoding='utf-8') as f:
                json.dump(texts, f, ensure_ascii=False, indent=4)
                print(f"Saved {len(texts)} items to the demo dataset after hallucination generation.")
        except Exception as e:
            raise Exception(f'Error saving dataset: {str(e)}')

# 处理完毕，输出结果
print("数据处理完成，已保存到demo.json文件中。")






# 金沸草具有抑菌和改善记忆的药理作用。抑菌作用是金沸草的主要药理效果之一，能有效抑制多种细菌的生长。同时，金沸草中的某些成分也被认为能够促进脑部血液循环，从而有助于改善记忆。