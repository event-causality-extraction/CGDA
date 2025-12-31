# !/usr/bin/env python3
"""
Reward Model类。

"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):

    def __init__(self, encoder):
        """
        init func.

        Args:
            encoder (transformers.AutoModel): backbone,
        """
        super().__init__()
        self.encoder = encoder
        self.reward_layer = nn.Linear(768, 1)# reward layer用于映射到1维reward

    def forward(
        self,
        input_ids: torch.tensor,
        token_type_ids: torch.tensor,
        attention_mask=None,
        pos_ids=None,
    ) -> torch.tensor:
        """
        forward 函数，返回每句话的得分值。

        Args:
            input_ids (torch.tensor): (batch, seq_len)
            token_type_ids (torch.tensor): (batch, seq_len)
            attention_mask (torch.tensor): (batch, seq_len)
            pos_ids (torch.tensor): (batch, seq_len)

        Returns:
            reward: (batch, 1)
        """
        pooler_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=pos_ids,
            attention_mask=attention_mask,
        )["pooler_output"]                              # (batch, hidden_size)
        reward = self.reward_layer(pooler_output)       # (batch, 1)
        return reward

class RewardModel_bart(nn.Module):

    def __init__(self, encoder):
        """
        init func.

        Args:
            encoder (transformers.AutoModel): backbone,
        """
        super().__init__()
        self.encoder = encoder
        self.reward_layer = nn.Linear(1024, 1)# reward layer用于映射到1维reward

    def forward(
        self,
        input_ids: torch.tensor,
        token_type_ids: torch.tensor,
        attention_mask=None,
        pos_ids=None,
    ) -> torch.tensor:
        """
        forward 函数，返回每句话的得分值。

        Args:
            input_ids (torch.tensor): (batch, seq_len)
            token_type_ids (torch.tensor): (batch, seq_len)
            attention_mask (torch.tensor): (batch, seq_len)
            pos_ids (torch.tensor): (batch, seq_len)

        Returns:
            reward: (batch, 1)
        """
        pooler_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=pos_ids,
            attention_mask=attention_mask,
        )["pooler_output"]                              # (batch, hidden_size)
        reward = self.reward_layer(pooler_output)       # (batch, 1)
        return reward
    
    
def compute_rank_list_loss(rank_rewards_list: List[List[torch.tensor]], device='cuda:0') -> torch.Tensor:
    """
    通过给定的有序（从高到低）的ranklist的reward列表，计算rank loss。
    所有排序高的句子的得分减去排序低的句子的得分差的总和，并取负。

    Args:
        rank_rewards_list (torch.tensor): 有序（从高到低）排序句子的reward列表，e.g. -> 
                                        [
                                            [torch.tensor([0.3588]), torch.tensor([0.2481]), ...],
                                            [torch.tensor([0.5343]), torch.tensor([0.2442]), ...],
                                            ...
                                        ]
        device (str): 使用设备
    
    Returns:
        loss (torch.tensor): tensor([0.4891], grad_fn=<DivBackward0>)
    """
    if type(rank_rewards_list) != list:
        raise TypeError(f'@param rank_rewards expected "list", received {type(rank_rewards)}.')

    
    loss, add_count = torch.tensor([float(0)], requires_grad=True).to(device), 0
    for rank_rewards in rank_rewards_list:
        for i in range(len(rank_rewards)-1):                             # 遍历所有前项-后项的得分差
            for j in range(i+1, len(rank_rewards)):
                diff = F.logsigmoid(rank_rewards[i] - rank_rewards[j])   # sigmoid到0~1之间
                loss = loss + diff
                add_count += 1
    loss = loss / add_count
    return -loss                                                               # 要最大化分差，所以要取负数


if __name__ == '__main__':
    # from rich import print
    from transformers import AutoModel, AutoTokenizer

    encoder = AutoModel.from_pretrained('/data01/zhanghang/PLM/bert-base')
    model = RewardModel(encoder)
    tokenizer = AutoTokenizer.from_pretrained('/data01/zhanghang/PLM/bert-base')

    batch_texts = [[
    "[sentence]By the late 1980s , however , the Bank was caught up in the debt crisis in the developing world caused by the recession and the dramatic rise in interest rates .[/sentence][words]['By', 'the', 'late', '1980s', ',', 'however', ',', 'the', 'Bank', 'was', 'caught', 'up', 'in', 'the', 'debt', 'crisis', 'in', 'the', 'developing', 'world', 'caused', 'by', 'the', 'recession', 'and', 'the', 'dramatic', 'rise', 'in', 'interest', 'rates', '.'][/words][label]['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Effect', 'I-Effect', 'E-Effect', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Cause', 'E-Cause', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'][/label]",
    "[sentence]By the late 1980s , however , the Bank was caught up in the debt crisis in the developing world caused by the recession and the dramatic rise in interest rates .[/sentence][words]['By', 'the', 'late', '1980s', ',', 'however', ',', 'the', 'Bank', 'was', 'caught', 'up', 'in', 'the', 'debt', 'crisis', 'in', 'the', 'developing', 'world', 'caused', 'by', 'the', 'recession', 'and', 'the', 'dramatic', 'rise', 'in', 'interest', 'rates', '.'][/words][label]['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I-Effect', 'O', 'O', 'O', 'B-Effect', 'I-Effect', 'E-Effect', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Cause', 'E-Cause', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'][/label]",
    "[sentence]By the late 1980s , however , the Bank was caught up in the debt crisis in the developing world caused by the recession and the dramatic rise in interest rates .[/sentence][words]['By', 'the', 'late', '1980s', ',', 'however', ',', 'the', 'Bank', 'was', 'caught', 'up', 'in', 'the', 'debt', 'crisis', 'in', 'the', 'developing', 'world', 'caused', 'by', 'the', 'recession', 'and', 'the', 'dramatic', 'rise', 'in', 'interest', 'rates', '.'][/words][label]['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Cause', 'I-Effect', 'E-Effect', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Effect', 'E-Cause', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'][/label]",
    "[sentence]By the late 1980s , however , the Bank was caught up in the debt crisis in the developing world caused by the recession and the dramatic rise in interest rates .[/sentence][words]['By', 'the', 'late', '1980s', ',', 'however', ',', 'the', 'Bank', 'was', 'caught', 'up', 'in', 'the', 'debt', 'crisis', 'in', 'the', 'developing', 'world', 'caused', 'by', 'the', 'recession', 'and', 'the', 'dramatic', 'rise', 'in', 'interest', 'rates', '.'][/words][label]['O', 'I-Effect', 'O', 'O', 'I-Effect', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I-Effect', 'E-Cause', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Cause', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'][/label]",
    "[sentence]By the late 1980s , however , the Bank was caught up in the debt crisis in the developing world caused by the recession and the dramatic rise in interest rates .[/sentence][words]['By', 'the', 'late', '1980s', ',', 'however', ',', 'the', 'Bank', 'was', 'caught', 'up', 'in', 'the', 'debt', 'crisis', 'in', 'the', 'developing', 'world', 'caused', 'by', 'the', 'recession', 'and', 'the', 'dramatic', 'rise', 'in', 'interest', 'rates', '.'][/words][label]['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Effect', 'I-Effect', 'E-Effect', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Cause', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'][/label]"
    ],
    [
    "[sentence]This case arises from a December 21 , 2005 automobile accident that resulted in the death of Larry Haynes .[/sentence][words]['This', 'case', 'arises', 'from', 'a', 'December', '21', ',', '2005', 'automobile', 'accident', 'that', 'resulted', 'in', 'the', 'death', 'of', 'Larry', 'Haynes', '.'][/words][label]['O', 'O', 'O', 'O', 'B-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'E-Cause', 'O', 'O', 'O', 'B-Effect', 'E-Effect', 'O', 'O', 'O', 'O'][/label]",
    "[sentence]This case arises from a December 21 , 2005 automobile accident that resulted in the death of Larry Haynes .[/sentence][words]['This', 'case', 'arises', 'from', 'a', 'December', '21', ',', '2005', 'automobile', 'accident', 'that', 'resulted', 'in', 'the', 'death', 'of', 'Larry', 'Haynes', '.'][/words][label]['O', 'O', 'O', 'O', 'B-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'E-Effect', 'O', 'O', 'O', 'I-Cause', 'I-Cause', 'O', 'O', 'O', 'O'][/label]",
    "[sentence]This case arises from a December 21 , 2005 automobile accident that resulted in the death of Larry Haynes .[/sentence][words]['This', 'case', 'arises', 'from', 'a', 'December', '21', ',', '2005', 'automobile', 'accident', 'that', 'resulted', 'in', 'the', 'death', 'of', 'Larry', 'Haynes', '.'][/words][label]['O', 'O', 'O', 'O', 'B-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'E-Effect', 'O', 'O', 'O', 'I-Cause', 'I-Cause', 'O', 'O', 'O', 'O'][/label]",
    "[sentence]This case arises from a December 21 , 2005 automobile accident that resulted in the death of Larry Haynes .[/sentence][words]['This', 'case', 'arises', 'from', 'a', 'December', '21', ',', '2005', 'automobile', 'accident', 'that', 'resulted', 'in', 'the', 'death', 'of', 'Larry', 'Haynes', '.'][/words][label]['O', 'I-Cause', 'O', 'O', 'B-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'E-Effect', 'O', 'O', 'O', 'B-Effect', 'E-Effect', 'O', 'O', 'O', 'O'][/label]",
    "[sentence]This case arises from a December 21 , 2005 automobile accident that resulted in the death of Larry Haynes .[/sentence][words]['This', 'case', 'arises', 'from', 'a', 'December', '21', ',', '2005', 'automobile', 'accident', 'that', 'resulted', 'in', 'the', 'death', 'of', 'Larry', 'Haynes', '.'][/words][label]['O', 'O', 'O', 'O', 'B-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'E-Cause', 'O', 'O', 'O', 'B-Effect', 'E-Effect', 'O', 'O', 'O', 'O'][/label]",
    "[sentence]This case arises from a December 21 , 2005 automobile accident that resulted in the death of Larry Haynes .[/sentence][words]['This', 'case', 'arises', 'from', 'a', 'December', '21', ',', '2005', 'automobile', 'accident', 'that', 'resulted', 'in', 'the', 'death', 'of', 'Larry', 'Haynes', '.'][/words][label]['O', 'O', 'O', 'O', 'I-Cause', 'I-Cause', 'O', 'I-Cause', 'I-Cause', 'I-Cause', 'O', 'O', 'O', 'O', 'O', 'E-Effect', 'O', 'O', 'O', 'O'][/label]"
    ]]

    rank_rewards = []
    for texts in batch_texts:
        tmp = []
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt')
            r = model(**inputs)
            tmp.append(r[0])
        rank_rewards.append(tmp)
    print('rank_rewards: ', rank_rewards)
    loss = compute_rank_list_loss(rank_rewards)
    print('loss: ', loss)
    # loss.backward()