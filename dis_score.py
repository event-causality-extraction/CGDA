# 数据特征打分与数据筛选函数
# 统计输入数据的文本长度和因果密度的均值和方差，对生成数据进行筛选
import json
import argparse
import numpy as np
from typing import List, Dict, Any


def read_json(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_to_json(output_data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)


def compute_alignment_scores(original_data: List[Dict], generated_data: List[Dict],
                             alpha: float = 0.5, beta: float = 0.5,
                             keep_ratio: float = 0.5, epsilon: float = 1e-8) -> List[Dict]:
    """
    对生成数据进行分布对齐打分，并返回带 score 的筛选后数据

    Args:
        original_data: 原始数据列表，每个元素包含 'sentence', 'cause', 'effect'
        generated_data: 生成数据列表，格式同上
        alpha: 文本长度得分权重 (默认 0.5)
        beta: 因果密度得分权重 (默认 0.5)
        keep_ratio: 保留前百分之多少的高分样本（默认 0.5 表示保留前 50%）
        epsilon: 小正数，避免除零

    Returns:
        筛选后的生成数据列表，每条数据增加 'score' 字段（按得分降序排列）
    """
    # Step 1: 计算原始数据的统计量
    lengths = []
    causality_densities = []

    for item in original_data:
        sentence = item['sentence']
        cause = item['cause']
        effect = item['effect']

        L_i = len(sentence)
        L_c = len(cause)
        L_e = len(effect)

        lengths.append(L_i)
        C_i = (L_c + L_e) / L_i if L_i > 0 else 0.0
        causality_densities.append(C_i)

    mu_l = np.mean(lengths)
    sigma_l = np.std(lengths)
    mu_c = np.mean(causality_densities)
    sigma_c = np.std(causality_densities)

    # Step 2: 为生成样本打分
    scored_generated = []
    for gen_item in generated_data:
        sentence = gen_item['sentence']
        cause = gen_item.get('cause', '')
        effect = gen_item.get('effect', '')

        L_g = len(sentence)
        L_c_g = len(cause)
        L_e_g = len(effect)
        C_g = (L_c_g + L_e_g) / L_g if L_g > 0 else 0.0

        S_L = max(0, 1 - abs(L_g - mu_l) / (sigma_l + epsilon))
        S_C = max(0, 1 - abs(C_g - mu_c) / (sigma_c + epsilon))

        score = alpha * S_L + beta * S_C

        gen_item_with_score = gen_item.copy()
        gen_item_with_score['score'] = round(score, 4)
        scored_generated.append(gen_item_with_score)

    # Step 3: 按得分排序并保留 top-k%
    scored_generated.sort(key=lambda x: x['score'], reverse=True)
    keep_n = int(len(scored_generated) * keep_ratio)
    return scored_generated[:keep_n]


def main():
    parser = argparse.ArgumentParser(
        description="基于文本长度与因果密度分布对生成数据进行打分与筛选"
    )
    parser.add_argument("--original_data", type=str, required=True,
                        help="原始训练数据 JSON 文件路径")
    parser.add_argument("--generated_data", type=str, required=True,
                        help="生成数据 JSON 文件路径")
    parser.add_argument("--output_path", type=str, required=True,
                        help="筛选后输出的 JSON 文件路径")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="文本长度得分的权重（默认: 0.5）")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="因果密度得分的权重（默认: 0.5）")
    parser.add_argument("--keep_ratio", type=float, default=0.5,
                        help="保留得分最高的比例（0.0~1.0，默认: 0.5 即 50%）")

    args = parser.parse_args()

    # 校验权重
    if not (0 <= args.alpha <= 1 and 0 <= args.beta <= 1):
        raise ValueError("alpha 和 beta 应在 [0, 1] 范围内")
    if abs(args.alpha + args.beta - 1.0) > 1e-6:
        print(f"注意：alpha + beta = {args.alpha + args.beta} ≠ 1，综合得分可能不归一化")

    if not (0 < args.keep_ratio <= 1.0):
        raise ValueError("keep_ratio 必须在 (0, 1] 范围内")

    # 加载数据
    original_data = read_json(args.original_data)
    generated_data = read_json(args.generated_data)

    # 打分与筛选
    filtered_data = compute_alignment_scores(
        original_data=original_data,
        generated_data=generated_data,
        alpha=args.alpha,
        beta=args.beta,
        keep_ratio=args.keep_ratio
    )

    # 保存结果
    save_to_json(filtered_data, args.output_path)
    print(f"筛选完成！共保留 {len(filtered_data)} 条数据，已保存至：{args.output_path}")


if __name__ == "__main__":
    main()

""""
python dis_score.py \
  --original_data /data01/hujunjie/CGDA_REBUTTAL/dataset/fincausal/doc/train_5.json \
  --generated_data /data01/hujunjie/CGDA_REBUTTAL/dataset/fincausal/gen_doc/train_5.json \
  --output_path /data01/hujunjie/CGDA_REBUTTAL/dataset/fincausal/rule_filter/train_5.json \
  --alpha 0.5 \
  --beta 0.5 \
  --keep_ratio 0.5
"""