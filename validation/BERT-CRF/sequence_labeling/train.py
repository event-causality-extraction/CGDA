# train.py
import argparse
import logging
import pickle
from pathlib import Path

import torch
import transformers
from torch.utils.tensorboard import SummaryWriter
from torchcrf import CRF
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AdamW, get_linear_schedule_with_warmup

from load import load_data
from model import LangModelWithDense
from utils import Meter, dump_args, init_logger, print_info

logger = logging.getLogger("bert.train")

def save_checkpoint(state, path):
    torch.save(state, path / "model.pt")

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = path / "model.pt"
    if ckpt.exists():
        state = torch.load(ckpt)
        model.load_state_dict(state['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        if scheduler:
            scheduler.load_state_dict(state['scheduler_state_dict'])
        return state.get('epoch', 0), state.get('best_f1', -1)
    return 0, -1

def train_model(args, model, train_loader, dev_loader, num_classes, target_classes, label_encoder, device):
    accum_steps = args.accum_steps if hasattr(args, "accum_steps") else 1
    train_meter, dev_meter = Meter(target_classes), Meter(target_classes)
    tb_writer = SummaryWriter(comment=f"-{args.run_name}" if args.run_name else "")
    train_losses, dev_losses = [], []
    patience, no_improve_epochs = 100, 0

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) // accum_steps * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    criterion = args.criterion
    start_epoch, best_f1 = load_checkpoint(args.save_path, model, optimizer, scheduler)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        model.train(); optimizer.zero_grad()
        for step, (x, y, mask, crf_mask) in enumerate(tqdm(train_loader), 1):
            logits = model(x, mask)
            loss = -criterion(logits.to(device), y, reduction="token_mean", mask=crf_mask) if args.crf else criterion(logits.reshape(-1, num_classes).to(device), y.reshape(-1).to(device))
            (loss / accum_steps).backward()
            if step % accum_steps == 0 or step == len(train_loader):
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            train_meter.update_params(loss.item(), logits, y)
        tb_writer.add_scalar("Train/loss", train_meter.loss, epoch)
        tb_writer.add_scalar("Train/macro_f1", train_meter.macro_f1, epoch)
        train_losses.append(train_meter.loss)
        train_meter.reset()

        model.eval()
        for x, y, mask, crf_mask in tqdm(dev_loader):
            logits = model(x, mask)
            loss = -criterion(logits.to(device), y, reduction="token_mean", mask=crf_mask) if args.crf else criterion(logits.reshape(-1, num_classes).to(device), y.reshape(-1).to(device))
            dev_meter.update_params(loss.item(), logits, y)
        tb_writer.add_scalar("Dev/loss", dev_meter.loss, epoch)
        tb_writer.add_scalar("Dev/macro_f1", dev_meter.macro_f1, epoch)
        dev_losses.append(dev_meter.loss)

        if dev_meter.macro_f1 > best_f1:
            best_f1 = dev_meter.macro_f1
            args.save_path.mkdir(parents=True, exist_ok=True)
            torch.save(model, args.save_path / "model.pt")
            with (args.save_path / "label_encoder.pk").open("wb") as f:
                pickle.dump(label_encoder, f)
            with (args.save_path / "best").open("w") as f:
                f.write(f"epoch: {epoch + 1} macro_f1: {best_f1}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping triggered."); break

        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_f1
        }, args.save_path)
        dev_meter.reset()

    tb_writer.close()
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(dev_losses)+1), dev_losses, label="Dev Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curve")
    plt.legend(); plt.grid(True)
    plt.savefig(args.save_path / "loss_curve.png")
    plt.close()

def main(args):
    device = torch.device(args.device)
    train_loader, dev_loader, label_encoder = load_data(
        args.train_path, args.dev_path, args.batch_size,
        args.tokens_column, args.predict_column,
        args.lang_model_name, args.max_len, args.separator,
        args.pad_label, args.null_label, device)

    base_model = transformers.AutoModel.from_pretrained(args.lang_model_name)
    input_size = 768 if "base" in args.lang_model_name else 1024
    model = LangModelWithDense(base_model, input_size, len(label_encoder.classes_), args.fine_tune).to(device)

    if args.crf:
        criterion = CRF(len(label_encoder.classes_), batch_first=True).to(device)
    else:
        weights = torch.tensor([
            1 if l not in [args.pad_label, args.null_label] else 0
            for l in label_encoder.classes_
        ], dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

    args.criterion = criterion
    classes = label_encoder.classes_.tolist()
    for special in [args.pad_label, args.null_label]: classes.remove(special)
    target_classes = [label_encoder.transform([c])[0] for c in classes]
    print_info(target_classes, label_encoder, args.lang_model_name, args.fine_tune, device)
    train_model(args, model, train_loader, dev_loader, len(label_encoder.classes_), target_classes, label_encoder, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path")
    parser.add_argument("dev_path")
    parser.add_argument("tokens_column", type=int)
    parser.add_argument("predict_column", type=int)
    parser.add_argument("lang_model_name")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save_path", type=Path, default="models")
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--separator", type=str, default=" ")
    parser.add_argument("--pad_label", type=str, default="<pad>")
    parser.add_argument("--null_label", type=str, default="<X>")
    parser.add_argument("--crf", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--logfile", type=str, default="train.log")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-all", action="store_true")
    parser.add_argument("--accum_steps", type=int, default=4)
    args = parser.parse_args()
    init_logger(args.logfile, None if args.log_all else "bert")
    dump_args(args)
    main(args)
