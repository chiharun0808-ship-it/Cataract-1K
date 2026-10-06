"""
手順6: TimeSformer を学習する（現行モデル 91.4% の再調査。条件: conditions/defectfree_f30_i333ms_s5s_vitprep.yaml）

学習の中身は、現行モデルを作ったコミット済みの training/train3.py（793d182）と同じにする。違いは次だけ:
- データ: lib/window_dataset.py（56症例の窓、元動画の PNG、事前学習と同じ前処理 lib/preprocess.py）
- 進め方: 50 エポックを再開なしで 1 回で通す（patience 10）。現行は 15+35 の 2 段階
- seed を固定する（現行は seed なし）
- 毎エポック、val の予測が何クラスに分かれているかを記録する（1 クラスへの崩壊の監視）

同じにするもの: TimeSformer（ViT-B/16 の空間方向の重みで初期化、dropout 0.1、attn_dropout 0）、Adam、lr 1e-5、
weight decay 0、batch 1 × 勾配累積 8、勾配クリップ 1.0、クラス重み付き CrossEntropy（重みの式は
dataset_weight/class_weight.py の get_class_weights と同じ。56症例の train の窓から計算）、データ拡張なし、
val の窓単位 macro F1 が最良のエポックを best_model.pth として保存。

実行（phase recognition dataset codes/ 直下で）:
    python experiments/planB_fixed_split/scripts/06_train_timesformer.py                  # 本番
    python experiments/planB_fixed_split/scripts/06_train_timesformer.py --smoke-steps 16 # 動作確認（数ステップだけ）
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.paths import CODES_DIR, CONDITIONS_DIR, PLAN_DIR, RUNS_DIR
from lib.window_dataset import WindowFrameDataset

sys.path.insert(0, str(CODES_DIR))
from dataset_weight.class_weight import get_class_weights  # noqa: E402  現行と同じ重みの式
from models.timesformer import create_timesformer_model  # noqa: E402

CONDITION_TAG = "defectfree_f30_i333ms_s5s_vitprep"
PHASES = [
    "Incision", "Viscoelastic", "Capsulorhexis", "Hydrodissection",
    "Phacoemulsification", "Irrigation_Aspiration", "CapsulePulishing",
    "LensImplantation", "LensPositioning", "Viscoelastic_Suction",
    "Anterior_ChamberFlushing", "Tonifying_Antibiotics", "idle",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PLAN_DIR, text=True).strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-steps", type=int, default=0,
                        help="動作確認用。>0 なら 1 エポック・train/val ともにこのバッチ数だけ回す")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    cond = yaml.safe_load(open(CONDITIONS_DIR / f"{CONDITION_TAG}.yaml", encoding="utf-8"))
    tr = cond["training"]
    seed = int(tr["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    smoke = args.smoke_steps > 0
    run_name = f"{date.today().isoformat()}_{CONDITION_TAG}_timesformer_seed{seed}" + ("_SMOKE" if smoke else "")
    run_dir = Path(args.run_dir) if args.run_dir else RUNS_DIR / run_name
    if run_dir.exists() and not smoke and any(run_dir.iterdir()):
        raise SystemExit(f"[停止] {run_dir} は既にあります。上書きしないため止めます。")
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(CONDITIONS_DIR / f"{CONDITION_TAG}.yaml", run_dir / "condition.yaml")

    window_csv = PLAN_DIR / cond["data"]["window_csv"]
    num_epochs = 1 if smoke else int(tr["num_epochs"])
    patience = int(tr["early_stopping_patience"])
    lr = float(tr["learning_rate"])
    accum = int(tr["grad_accum_steps"])
    clip = float(tr["grad_clip_max_norm"])
    seq_len = int(cond["input"]["sequence_length"])

    class_weights = get_class_weights(csv_path=str(window_csv)).to(device)
    train_ds = WindowFrameDataset(window_csv, "train")
    val_ds = WindowFrameDataset(window_csv, "val")
    g = torch.Generator()
    g.manual_seed(seed)
    loader_kw = dict(batch_size=int(tr["batch_size"]), num_workers=8, persistent_workers=True,
                     pin_memory=(device.type == "cuda"), worker_init_fn=seed_worker)
    train_loader = DataLoader(train_ds, shuffle=True, generator=g, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = create_timesformer_model("timesformer", num_classes=len(PHASES), num_frames=seq_len,
                                     pretrained_cnn=True, dropout=float(tr["dropout"]),
                                     attn_dropout=float(tr["attn_dropout"])).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(tr["label_smoothing"]))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=float(tr["weight_decay"]))

    config = {
        "condition_tag": CONDITION_TAG, "started_at": datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(), "device": str(device), "torch": torch.__version__,
        "n_train_windows": len(train_ds), "n_val_windows": len(val_ds),
        "class_weights": [round(float(x), 6) for x in class_weights.cpu()],
        "num_epochs": num_epochs, "patience": patience, "learning_rate": lr, "grad_accum_steps": accum,
        "grad_clip_max_norm": clip, "seed": seed, "smoke_steps": args.smoke_steps,
    }
    json.dump(config, open(run_dir / "config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(config, ensure_ascii=False, indent=2))

    best_f1, stale = -1.0, 0
    history = []
    log = open(run_dir / "training_log.txt", "a", encoding="utf-8", buffering=1)
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        tr_loss, tr_preds, tr_labels = 0.0, [], []
        n_batches = len(train_loader) if not smoke else min(args.smoke_steps, len(train_loader))
        for i, (x, y) in enumerate(train_loader):
            if i >= n_batches:
                break
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y) / accum
            loss.backward()
            if (i + 1) % accum == 0 or (i + 1) == n_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
                optimizer.step()
                optimizer.zero_grad()
            tr_loss += loss.item() * accum
            tr_preds += out.argmax(1).cpu().tolist()
            tr_labels += y.cpu().tolist()
            if (i + 1) % 200 == 0:
                print(f"  epoch {epoch} batch {i + 1}/{n_batches}", flush=True)

        model.eval()
        va_loss, va_preds, va_labels = 0.0, [], []
        n_val = len(val_loader) if not smoke else min(args.smoke_steps, len(val_loader))
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                if i >= n_val:
                    break
                x, y = x.to(device), y.to(device)
                out = model(x)
                va_loss += criterion(out, y).item()
                va_preds += out.argmax(1).cpu().tolist()
                va_labels += y.cpu().tolist()

        tr_f1 = f1_score(tr_labels, tr_preds, average="macro", zero_division=0)
        va_f1 = f1_score(va_labels, va_preds, average="macro", zero_division=0)
        n_pred_classes = len(set(va_preds))
        rec = {"epoch": epoch, "train_loss": tr_loss / max(n_batches, 1), "val_loss": va_loss / max(n_val, 1),
               "train_f1": tr_f1, "val_f1": va_f1,
               "train_acc": float(np.mean(np.array(tr_preds) == np.array(tr_labels))),
               "val_acc": float(np.mean(np.array(va_preds) == np.array(va_labels))),
               "val_n_predicted_classes": n_pred_classes, "time": datetime.now().isoformat(timespec="seconds")}
        history.append(rec)
        msg = (f"Epoch {epoch}/{num_epochs} | Loss {rec['train_loss']:.4f} | Train F1 {tr_f1:.4f} | "
               f"Val F1 {va_f1:.4f} | val の予測クラス数 {n_pred_classes}")
        if n_pred_classes <= 2:
            msg += "  【注意】予測が少数のクラスに偏っている（崩壊の疑い）"
        print(msg, flush=True)
        log.write(msg + "\n")
        json.dump(history, open(run_dir / "history.json", "w", encoding="utf-8"), indent=2)

        if va_f1 > best_f1:
            best_f1, stale = va_f1, 0
            torch.save(model.state_dict(), run_dir / "best_model.pth")
            cm = confusion_matrix(va_labels, va_preds, labels=range(len(PHASES)))
            np.savetxt(run_dir / "best_val_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
            log.write(f"  [best 更新] epoch {epoch} val F1 {va_f1:.4f}\n")
        else:
            stale += 1
            if stale >= patience:
                log.write(f"Early stopping: {patience} エポック改善なし（epoch {epoch}）\n")
                print("Early stopping", flush=True)
                break

    torch.save(model.state_dict(), run_dir / "last_model.pth")
    summary = {"best_val_f1": best_f1, "best_epoch": max(history, key=lambda r: r["val_f1"])["epoch"],
               "epochs_run": len(history), "finished_at": datetime.now().isoformat(timespec="seconds")}
    json.dump(summary, open(run_dir / "train_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("完了:", summary)


if __name__ == "__main__":
    main()
