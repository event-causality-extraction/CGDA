# 读取原始训练数据，采样出指定百分比的数据
import random
import argparse
from typing import List

def read_txt_blocks(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = [block.strip() for block in content.split('\n\n') if block.strip()]
    return blocks


def sample_blocks(blocks: List[str], k: float, seed: int = None) -> List[str]:
    if seed is not None:
        random.seed(seed)
    
    n_total = len(blocks)
    n_sample = max(1, int(round(n_total * k / 100))) if k > 0 else 0
    n_sample = min(n_sample, n_total)
    
    sampled = random.sample(blocks, n_sample)
    return sampled


def write_sampled_txt(blocks: List[str], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(blocks) + '\n')


def sample_multiple_ratios(input_file: str, k_list: List[float], output_prefix: str, seed: int):

    blocks = read_txt_blocks(input_file)
    
    for k in k_list:
        sampled_blocks = sample_blocks(blocks, k, seed=seed)
        output_file = f"{output_prefix}_{k}.txt"
        write_sampled_txt(sampled_blocks, output_file)
        print(f"已保存 {k}% 采样数据到 {output_file}（共 {len(sampled_blocks)} 条）")


def main():
    parser = argparse.ArgumentParser(description="对文本数据按比例采样并保存多个子集")
    parser.add_argument("--input_file", type=str, required=True,
                        help="输入的原始训练数据文件路径（以空行分隔样本）")
    parser.add_argument("--ratios", type=float, nargs='+', required=True,
                        help="要采样的百分比，例如：1 2 3 5 7 10")
    parser.add_argument("--output_prefix", type=str, default="sampled",
                        help="输出文件名前缀，默认为 'sampled'")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，默认为 42")

    args = parser.parse_args()

    sample_multiple_ratios(
        input_file=args.input_file,
        k_list=args.ratios,
        output_prefix=args.output_prefix,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

"""
python sample.py \
  --input_file /data01/hujunjie/CGDA_REBUTTAL/dataset/fincausal/txt/train_bieo.txt \
  --ratios 1 2 3 5 7 10 \
  --output_prefix /data01/hujunjie/CGDA_REBUTTAL/dataset/fincausal/txt/train \
  --seed 2025
"""