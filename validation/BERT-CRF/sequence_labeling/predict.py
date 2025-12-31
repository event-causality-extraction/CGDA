import argparse
import logging
import pickle
from pathlib import Path

import torch
import transformers
from tqdm import tqdm

from load import load_data_from_file
from utils import dump_args, init_logger

logger = logging.getLogger("bert.predict")


class LangModelWithDense(torch.nn.Module):
    def __init__(self, lang_model, emb_size, num_classes, fine_tune):
        super().__init__()
        self.num_classes = num_classes
        self.fine_tune = fine_tune

        self.lang_model = lang_model
        self.linear = torch.nn.Linear(emb_size, num_classes)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, mask):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        if not self.fine_tune:
            with torch.no_grad():
                self.lang_model.eval()
                outputs = self.lang_model(x, attention_mask=mask)
                embeddings = outputs.last_hidden_state
        else:
            outputs = self.lang_model(x, attention_mask=mask)
            embeddings = outputs.last_hidden_state

        logits = torch.zeros((batch_size, seq_len, self.num_classes), device=x.device)

        for i in range(seq_len):
            logit = self.dropout(self.linear(embeddings[:, i, :]))
            logits[:, i, :] = logit

        return logits

    def predict_with_confidence(self, x, mask):
        """
        返回每个 token 的预测类别和对应置信度（非标量）
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, mask)
            probs = torch.softmax(logits, dim=-1)              # shape: [B, L, C]
            max_probs, predictions = torch.max(probs, dim=-1)  # shape: [B, L]

        self.train()
        return predictions.cpu().tolist(), max_probs.cpu().tolist()


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    args.output_path.mkdir(exist_ok=True, parents=True)

    with (args.model_path / "label_encoder.pk").open("rb") as file:
        label_encoder = pickle.load(file)

    test_loader, _ = load_data_from_file(
        args.test_path,
        1,
        args.token_column,
        args.predict_column,
        args.lang_model_name,
        512,
        args.separator,
        args.pad_label,
        args.null_label,
        device,
        label_encoder,
        False,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.lang_model_name, use_fast=True
    )

    # 构建模型结构
    config = transformers.AutoConfig.from_pretrained(args.lang_model_name)
    lang_model = transformers.AutoModel.from_pretrained(args.lang_model_name, config=config)

    num_classes = len(label_encoder.classes_)
    emb_size = lang_model.config.hidden_size

    model = LangModelWithDense(lang_model, emb_size, num_classes, fine_tune=False)

    # 加载 checkpoint 并提取模型权重
    checkpoint = torch.load(args.model_path / "model.pt", map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    list_labels = []
    list_confidences = []

    logger.info("Predicting tags with confidence")
    for test_x, _, mask, _ in tqdm(test_loader):
        preds, confidences = model.predict_with_confidence(test_x, mask)

        length = mask.sum(dim=1).item() - 2  # 去掉 [CLS] 和 [SEP]

        labels = label_encoder.inverse_transform(preds[0][1:length+1])         # shape: [L']
        confidences_list = [round(float(c), 2) for c in confidences[0][1:length+1]]  # shape: [L']

        list_labels.append(labels)
        list_confidences.append(confidences_list)

    in_path = args.test_path
    out_path = args.output_path / args.output_name
    with in_path.open() as in_file, out_path.open("w") as out_file:
        sentence_idx = 0
        label_idx = 0

        for line in in_file:
            if line.startswith("#"):
                out_file.write(line)
            elif line.strip() == "":
                assert label_idx == len(list_labels[sentence_idx]), \
                    f"label_idx={label_idx}, expected={len(list_labels[sentence_idx])}"

                out_file.write("\n")
                sentence_idx += 1
                label_idx = 0
            else:
                tokens = line.strip().split(args.separator)

                token = tokens[args.token_column]
                gold = tokens[args.predict_column]

                try:
                    pred = list_labels[sentence_idx][label_idx]
                    confidence = list_confidences[sentence_idx][label_idx]
                except IndexError:
                    logger.error(f"IndexError at sentence {sentence_idx}, label_idx={label_idx}")
                    raise

                out_file.write(f"{token}{args.separator}{gold}{args.separator}{pred}{args.separator}{confidence}\n")

                # 使用 tokenizer 分词，更新 label_idx
                subtokens = tokenizer.tokenize(token)
                label_idx += len(subtokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_path", type=Path)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("token_column", type=int)
    parser.add_argument("predict_column", type=int)
    parser.add_argument("lang_model_name", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_path", type=Path, default="output")
    parser.add_argument("--output_name", type=str, default="predict_with_confidence.conllu")
    parser.add_argument("--separator", type=str, default=" ")
    parser.add_argument("--pad_label", type=str, default="<pad>")
    parser.add_argument("--null_label", type=str, default="<X>")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--log-all",
        action="store_true",
        help="Enable logging of everything, including libraries like transformers",
    )

    args = parser.parse_args()

    log_name = None if args.log_all else "bert"
    init_logger(log_name=log_name)
    dump_args(args)

    main(args)