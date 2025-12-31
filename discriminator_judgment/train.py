# Using the Reward Model approach from ChatGPT to train an adjudicator.
# Since it is a discriminative model, use the BERT (rather than GPT) model for training.

import os
import sys
import time
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, default_data_collator, get_scheduler

from reward_model import RewardModel, compute_rank_list_loss
from utils import convert_example
from iTrainingLogger import iSummaryWriter
import json

def parse_args():
    parser = argparse.ArgumentParser(description='训练脚本')
    
    # 添加一个必需的位置参数，用于指定配置文件路径
    parser.add_argument('config', type=str, help='配置文件的路径 (JSON 格式)')
    
    # 首先解析仅包含配置文件路径的参数
    args, remaining_argv = parser.parse_known_args()
    
    config_path = args.config
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"配置文件未找到: {config_path}")
        sys.exit(1)
    
    # 加载配置文件
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 基于配置文件中的参数动态添加命令行参数
    for key, value in config.items():
        arg_type = type(value)
        
        # 对于布尔类型，需要特殊处理
        if isinstance(value, bool):
            if value:
                parser.add_argument(f'--{key}', dest=key, action='store_false', help=f'将 {key} 设置为 False')
            else:
                parser.add_argument(f'--{key}', dest=key, action='store_true', help=f'将 {key} 设置为 True')
        else:
            parser.add_argument(f'--{key}', type=arg_type, default=value, help=f'覆盖配置文件中的 {key} 参数 (default: {value})')
    
    # 重新解析所有参数，包括命令行中的覆盖参数
    args = parser.parse_args()
    
    return args

# Create a logger for recording metrics and results during the training process.
args = parse_args()
writer = iSummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)

def evaluate_model(model, data_loader):
    """
    Evaluate the training performance of the current model on the test set.

    Args:
        model: The current model
        data_loader: Dataloader for the test set
    """
    model.eval()
    with torch.no_grad():
        batch_rank_rewards = []
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
                    # Compute the reward value for each sample
                    rank_rewards.append(reward[0])  # (rank_text_num) -> [tensor([0.1696]), tensor([0.3466])]
                # Add rank_rewards to the batch_rank_rewards list to form a 2D array
                batch_rank_rewards.append(
                    rank_rewards)  # (batch, rank_text_num) -> [[tensor([0.1696]), tensor([0.3466])], ...]
    model.train()
    total_ranklist, right_ranklist = 0, 0
    for rank_rewards in batch_rank_rewards:
        rank_rewards = [t.cpu().float() for t in rank_rewards]
        rank_rewards_sorted = sorted(rank_rewards, reverse=True)
        total_ranklist += 1
        if rank_rewards_sorted == rank_rewards:
            right_ranklist += 1
    return right_ranklist / total_ranklist

def train():
    encoder = AutoModel.from_pretrained(args.model)
    model = RewardModel(encoder=encoder)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset('text', data_files={'train': args.train_path,
                                               'dev': args.dev_path},
                                               )
    print(dataset)
    convert_func = partial(convert_example, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    dataset = dataset.map(convert_func, batched=True)

    train_dataset = dataset["train"]
    dev_dataset = dataset["dev"]
    train_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=default_data_collator,
                                  batch_size=args.batch_size)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, collate_fn=default_data_collator,
                                 batch_size=args.batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    model.to(args.device)

    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    warm_steps = int(args.warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )

    loss_list = []
    tic_train = time.time()
    global_step, best_acc = 0, 0

    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, 'model_best.pt')

    for epoch in range(1, args.num_train_epochs + 1):
        for batch in train_dataloader:
            batch_rank_rewards = []
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
                    rank_rewards.append(reward[0])  # (rank_text_num) -> [tensor([0.1696]), tensor([0.3466])]
                batch_rank_rewards.append(
                    rank_rewards)  # (batch, rank_text_num) -> [[tensor([0.1696]), tensor([0.3466])], ...]
            loss = compute_rank_list_loss(batch_rank_rewards, device=args.device)
            # if torch.isnan(loss).any():
            #     print("Loss became NaN!")
            # # 打印相关信息，如当前批次的数据、模型参数等
            #     break  # 或者采取其他措施
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_list.append(float(loss.cpu().detach()))

            global_step += 1
            if global_step % args.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                writer.add_scalar('train/train_loss', loss_avg, global_step)
                print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                      % (global_step, epoch, loss_avg, args.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % args.valid_steps == 0:
                # 仅在验证时保存最佳模型
                acc = evaluate_model(model, dev_dataloader)
                print(f"Validation Accuracy: {acc:.5f}")
                writer.add_scalar('eval/accuracy', acc, global_step)
                writer.record()
                print("Evaluation acc: %.5f" % (acc))
                if acc > best_acc:
                    print(
                        f"Best accuracy improved: {best_acc:.5f} --> {acc:.5f}"
                    )
                    best_acc = acc
                    try:
                        # 保存模型状态字典
                        torch.save(model.state_dict(), best_model_path)
                        # 保存 tokenizer
                        tokenizer.save_pretrained(args.save_dir)
                        print(f"最佳模型已保存到 {best_model_path}")
                    except Exception as e:
                        print(f"保存最佳模型时出错: {e}")
                tic_train = time.time()
                writer.record()

    print(f"训练完成。最佳验证准确率: {best_acc:.5f}")
    print(f"最佳模型已保存到 {best_model_path}")

if __name__ == '__main__':
    from rich import print
    train()
