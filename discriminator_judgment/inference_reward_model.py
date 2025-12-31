#!/usr/bin/env python3
# Use the trained adjudicator to annotate large and small models
# Output annotations for choosing large or small models into a TSV file
# python inference_reward_model.py

import torch
from rich import print
from transformers import AutoTokenizer, AutoModel, default_data_collator
from datasets import load_dataset
from functools import partial
from utils import convert_example
from torch.utils.data import DataLoader
import argparse
from reward_model import RewardModel
import csv
import os

def evaluate_model(model, data_loader):
    """
    Evaluate the training performance of the current model on the test set.

    Args:
        model: The current model
        data_loader: Dataloader for the test set
    """
    model.eval()
    batch_rank_rewards = []
    with torch.no_grad():
        for batch in data_loader:
            for batch_idx in range(len(batch['input_ids'])):
                rank_texts_count = len(batch['input_ids'][batch_idx])
                rank_rewards = []
                for text_idx in range(rank_texts_count):
                    reward = model(
                        batch['input_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
                        batch['token_type_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
                        batch['attention_mask'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
                        batch['position_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
                    )
                    # 计算每个样本的奖励值
                    rank_rewards.append(reward[0])  # (rank_text_num) -> [tensor([0.1696]), tensor([0.3466])]
                # 将 rank_rewards 添加到 batch_rank_rewards 列表中形成二维数组
                batch_rank_rewards.append(rank_rewards)  # (batch, rank_text_num) -> [[tensor(...), ...], ...]
    
    total_ranklist, right_ranklist = 0, 0
    for rank_rewards in batch_rank_rewards:
        rank_rewards = [t.cpu().float() for t in rank_rewards]
        rank_rewards_sorted = sorted(rank_rewards, reverse=True)
        total_ranklist += 1
        if rank_rewards_sorted == rank_rewards:
            right_ranklist += 1
    accuracy = right_ranklist / total_ranklist if total_ranklist > 0 else 0
    return accuracy, batch_rank_rewards

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--device', default="cuda:0", help="Select which device to train model, defaults to GPU.")
parser.add_argument("--max_seq_len", default=512, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences longer "
                         "than this will be truncated, sequences shorter will be padded.")
args = parser.parse_args()

# dataset = ["scite","fincausal","fcr"]
dataset = ["fcr"]

fewshot = ["001","003","005"]
for dataset_name in dataset:
    for fc in fewshot:
        model_path = f'/data01/hujunjie/Round2/model/RL-Model/{dataset_name}/rm-{fc}/model_best.pt'
        tokenizer = AutoTokenizer.from_pretrained(f'/data01/hujunjie/Round2/model/RL-Model/{dataset_name}/rm-{fc}/')
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

        # 重新加载模型以匹配词汇表大小
        encoder = AutoModel.from_pretrained('/data01/hujunjie/PLM_MODEL/bert-base-cased',ignore_mismatched_sizes=True)
        model = RewardModel(encoder=encoder)
        model.load_state_dict(torch.load(model_path, map_location=args.device), strict=False)
        model.to(args.device)
        model.eval()

        # 设置路径
        # output_dir = f'/data01/hujunjie/Round2/Result/RL_Model_Filter_Gen_data_result/{dataset_name}'
        # test_path = f"/data01/hujunjie/Round2/Result/RL_Model_Filter_Gen_data_result/{dataset_name}/filter_rl_result-{fc}.tsv"
        output_dir = f'/data01/hujunjie/Round2/Result/RL_Model_Gen_data_result/{dataset_name}'
        test_path = f"/data01/hujunjie/Round2/Result/RL_Model_Gen_data_result/{dataset_name}/filter_rl_result-{fc}.tsv"
        accuracy_file_path = os.path.join(output_dir, f'rl_result{fc}.tsv')

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 确保输入文件存在
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"测试文件未找到: {test_path}")

        # 1. 读取 TSV 文件，并手动指定列名
        dataset = load_dataset(
            'csv',
            data_files={'test': test_path},
            delimiter='\t',
            column_names=['sentence_id', 'word_index', 'llm_data', 'slm_data'],
            header=None  # 指定没有标题行
        )

        # 2. 仅使用 'input_data' 和 'output_data' 作为模型输入
        def process_input_data(example):
            # 如果模型需要将 'input_data' 和 'output_data' 结合起来作为输入，可以按需要修改
            combined_text = example['llm_data'] + "\t" + example['slm_data']
            return {'text': combined_text}

        dataset = dataset.map(process_input_data, remove_columns=['sentence_id', 'word_index', 'llm_data', 'slm_data'])

        # 3. 使用 convert_example 进行编码
        convert_func = partial(convert_example, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
        dataset = dataset.map(convert_func, batched=True)
        test_dataset = dataset['test']

        # 4. 创建 DataLoader
        test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=16)

        # 5. 评估模型
        acc, reward = evaluate_model(model, test_dataloader)

        # 6. 获取模型的预测结果（例如，选择概率最高的索引）
        max_values_and_indices = []

        for row in reward:
            max_tensor = max(row)  # 找到最大值的 tensor
            max_index = row.index(max_tensor)  # 确定最大值的索引
            max_values_and_indices.append(max_index)

        print("Accuracy:", acc)

        # 7. 将 source_id、index 和模型预测的结果保存到 TSV 文件中
        # 重新读取输入文件以获取 source_id 和 index
        with open(test_path, 'r', encoding='utf-8') as infile, open(accuracy_file_path, 'w', newline='', encoding='utf-8') as outfile:
            reader = csv.reader(infile, delimiter='\t')
            writer = csv.writer(outfile, delimiter='\t')
            
            # 写入标题行
            writer.writerow(['sentence_id', 'index', 'predicted_index'])
            
            for i, row in enumerate(reader):
                if len(row) < 4:
                    # 跳过格式不正确的行
                    continue
                source_id = row[0]
                index = row[1]
                predicted_index = max_values_and_indices[i] if i < len(max_values_and_indices) else ''
                writer.writerow([source_id, index, predicted_index])

        # 8. 将整体准确率保存到同一个 TSV 文件中（附加在结果文件后）
        with open(accuracy_file_path, 'a', encoding='utf-8') as acc_file:
            acc_file.write(f'\nAccuracy\t{acc}\n')