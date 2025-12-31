# 调用 API 生成数据
import json
import argparse
from openai import OpenAI
from tqdm import tqdm

# ==================== 配置区域 ====================
API_KEY = "your api key"
BASE_URL = "api url"
MODEL_NAME = "model name"

SYSTEM_PROMPT = """You are an Event Causality Data Generation assistant.
Your task is to generate data within the same Domain and Causality Type as the demonstration.
Each data item must include a Cause text and an Effect text, with no causal cue words in either."""
# =================================================


def read_json(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_to_json(output_data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)


def data_generation(input_data):
    output_data = []
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    for i in tqdm(range(len(input_data)), desc="生成进度"):
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(input_data[i])}
            ],
            model=MODEL_NAME
        )
        output_data.append(chat_completion.choices[0].message.content)

    return output_data


def main():
    parser = argparse.ArgumentParser(description="调用大模型生成事件因果抽取数据")
    parser.add_argument("--input_path", type=str, required=True,
                        help="输入 JSON 文件路径")
    parser.add_argument("--output_path", type=str, required=True,
                        help="输出 JSON 文件路径")

    args = parser.parse_args()

    input_data = read_json(args.input_path)
    output_data = data_generation(input_data)
    save_to_json(output_data, args.output_path)

    print(f"生成完成，生成数据已保存至文件：{args.output_path}")


if __name__ == "__main__":
    main()

"""
python generate_data.py \
  --input_path ../dataset/fincausal/doc/train.json \
  --output_path ../dataset/fincausal/gen_doc/train.json
"""