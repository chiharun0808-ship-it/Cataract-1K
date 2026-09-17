"""
スライディングウィンドウ方式のクリップ（build_sliding_window_index.py が生成した
dataset_index_sliding_window.csv）で学習するスクリプト。

train2.py との違いはここだけ:
  1. 参照するインデックスCSVが dataset_index_sliding_window.csv になる
  2. SurgicalPhaseDataset に frame_interval=1 を渡す
     （build_sliding_window_index.py が書き出すクリップは、既に学習時と同じ1/3秒間隔で
     サンプリング済みの30枚なので、ここで再度間引く必要がない。frame_interval=10のままだと
     30枚中3枚しか読まず残り27枚をパディングで埋めてしまう事故になるので注意）
  3. 保存先ディレクトリ名に sliding_window_ prefixを付け、クリップ単位学習の結果と
     混ざらないようにする

それ以外（モデル選択・クラス重み付け損失・交差検証・学習ループ・ログ/混同行列の保存形式）は
train2.py と共通化しており、そちらの知識がそのまま使える。

評価は2本立て（ユーザーと合意した方針）:
  - クリップ単位のtest評価: 既存の test2.py がそのまま使える
    （checkpoint_dir の config.json から csv_path を読むが、config.json に記録される
    csv_path は学習に使ったCSVパスなので、test2.py で評価する際は以下のように
    --csv-path 的な上書きが必要 -> 現状の test2.py は csv_path を config.json から固定で
    読むため、クリップ単位のtest評価には従来通り dataset_index_split.csv のtest split
    （両CSVでcase単位のsplitは共通）を使う。test2.py 自体の変更は不要。
  - 動画1本を流した毎秒一致率（ストリーミング評価）: 既存の visualize_phase_on_video.py を
    test split の各症例動画に対して --gt-csv 付きで実行すればそのまま得られる
    （stride_sec実験で使ったのと同じ使い方）。

使い方:
    python train3.py
"""
import gc
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from dataset_index.prepare_dataset import SurgicalPhaseDataset
from models.cnn_rnn_hybrids import create_cnn_rnn_model, CNN_RNN_MODELS
from models.timesformer import create_timesformer_model, TIMESFORMER_MODELS
from dataset_weight.class_weight import get_class_weights as get_class_weights_13
from dataset_weight.class_weight_without_idle import get_class_weights as get_class_weights_12
from pathlib import Path
from datetime import datetime

phases = [
    "Incision", "Viscoelastic", "Capsulorhexis", "Hydrodissection",
    "Phacoemulsification", "Irrigation_Aspiration", "CapsulePulishing",
    "LensImplantation", "LensPositioning", "Viscoelastic_Suction",
    "Anterior_ChamberFlushing", "Tonifying_Antibiotics", "idle"
]
phases_12 = [
    "Incision", "Viscoelastic", "Capsulorhexis", "Hydrodissection",
    "Phacoemulsification", "Irrigation_Aspiration", "CapsulePulishing",
    "LensImplantation", "LensPositioning", "Viscoelastic_Suction",
    "Anterior_ChamberFlushing", "Tonifying_Antibiotics"
]

# --- 1. 設定項目（train2.pyと同じ意味）---
INCLUDE_IDLE_PHASE = True
num_classes = 13 if INCLUDE_IDLE_PHASE else 12
active_phases = phases if INCLUDE_IDLE_PHASE else phases_12

USE_CROSS_VALIDATION = False
NUM_FOLDS = 5

MODEL_NAMES = ["timesformer"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# build_sliding_window_index.py の出力。train2.pyのdataset_index_split.csvとは
# 別ファイル（クリップの実体が別物のため）だが、case_id単位のsplit/foldの割り当ては
# dataset_index_split.csv と共通（build_sliding_window_index.pyがそこから引き継いでいる）。
csv_path = "data/dataset_index_sliding_window_stride5.csv"
batch_size = 4
sequence_length = 30
# build_sliding_window_index.py が書き出すクリップは既にサンプリング済みの30枚なので、
# SurgicalPhaseDataset側では間引かずそのまま読む（train2.pyのfrom_interval=10とは異なる）。
FRAME_INTERVAL = 1
# train2.pyはnum_workers=0（クリップ1,120個なら十分速かった）だが、スライディング
# ウィンドウ版はクリップ数が約4.5倍(5,082個)に増えており、単一プロセスでの
# クリップ読み込み(cv2でのフレーム読み出し)がボトルネックになる。CPUコア数に余裕が
# あるため、DataLoaderのワーカープロセスで読み込みを並列化しGPU待ちを減らす。
NUM_WORKERS = 8

MODEL_BATCH_SIZE_OVERRIDES = {
    "efficientnetb5_lstm": 1,
    "efficientnetb5_gru": 1,
    "timesformer": 1,
}

learning_rate = 1e-4
# 第一段階(num_epochs=15, patience=5)はearly stoppingが発動せず、val F1は
# 0.4957→0.7630まで15epoch目の時点でまだ右肩上がりのまま終了した。収束前に
# 打ち切った形なので、続きから長めに学習を延長する。
num_epochs = 35
patience = 10

# 第一段階のbest_model.pthから続きを学習する。Noneなら通常通りゼロから学習する。
# ここから再開することで、既に学習済みの15epoch分の計算を無駄にしない。
RESUME_FROM = Path("result/epochs_15/sliding_window_checkpoints_0913_21_46/best_model.pth")
# 上のチェックポイントの実際のval F1（training_log.txtのEpoch 15の値）。
# best_f1の初期値をこれにしておかないと、再開直後にそれより低いval F1でも
# 「ベスト更新」として誤って上書き保存してしまう。
RESUME_BEST_F1 = 0.7630

MODEL_LEARNING_RATE_OVERRIDES = {
    "timesformer": 1e-5,
}
MODEL_GRAD_ACCUM_STEPS = {
    "timesformer": 8,
}
GRAD_CLIP_MAX_NORM = 1.0

current_time = datetime.now().strftime("%m%d_%H_%M")


def run_training(save_dir: Path, model_name: str, fold=None) -> float:
    """train2.py の run_training と同じ構成。差分はコメント参照。"""
    os.makedirs(save_dir, exist_ok=True)

    effective_batch_size = MODEL_BATCH_SIZE_OVERRIDES.get(model_name, batch_size)
    effective_learning_rate = MODEL_LEARNING_RATE_OVERRIDES.get(model_name, learning_rate)
    grad_accum_steps = MODEL_GRAD_ACCUM_STEPS.get(model_name, 1)

    config = {
        "model_name": model_name,
        "training_scheme": "sliding_window",
        "include_idle_phase": INCLUDE_IDLE_PHASE,
        "num_classes": num_classes,
        "csv_path": csv_path,
        "frame_interval": FRAME_INTERVAL,
        "num_workers": NUM_WORKERS,
        "batch_size": effective_batch_size,
        "sequence_length": sequence_length,
        "learning_rate": effective_learning_rate,
        "grad_accum_steps": grad_accum_steps,
        "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
        "num_epochs": num_epochs,
        "patience": patience,
        "use_cross_validation": USE_CROSS_VALIDATION,
        "fold": fold,
        "num_folds": NUM_FOLDS if USE_CROSS_VALIDATION else None,
        "resumed_from": str(RESUME_FROM) if RESUME_FROM else None,
        "resume_best_f1": RESUME_BEST_F1 if RESUME_FROM else None,
    }
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    class_weights = (get_class_weights_13(csv_path=csv_path) if INCLUDE_IDLE_PHASE
                      else get_class_weights_12(csv_path=csv_path)).to(device)

    # --- 2. データセットの準備 ---
    train_dataset = SurgicalPhaseDataset(
        csv_path, split='train', fold=fold, sequence_length=sequence_length,
        frame_interval=FRAME_INTERVAL, exclude_idle=not INCLUDE_IDLE_PHASE,
    )
    val_dataset = SurgicalPhaseDataset(
        csv_path, split='val', fold=fold, sequence_length=sequence_length,
        frame_interval=FRAME_INTERVAL, exclude_idle=not INCLUDE_IDLE_PHASE,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=effective_batch_size, shuffle=True,
        num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=effective_batch_size, shuffle=False,
        num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=(device.type == "cuda"),
    )

    # --- 3. モデル・損失・最適化 ---
    if model_name in CNN_RNN_MODELS:
        model = create_cnn_rnn_model(model_name, num_classes=num_classes, pretrained_cnn=True).to(device)
    elif model_name in TIMESFORMER_MODELS:
        model = create_timesformer_model(
            model_name, num_classes=num_classes, num_frames=sequence_length, pretrained_cnn=True
        ).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=effective_learning_rate)

    if RESUME_FROM:
        model.load_state_dict(torch.load(RESUME_FROM, map_location=device))
        print(f"[sliding_window] {RESUME_FROM} から再開（読み込み時点のval F1: {RESUME_BEST_F1:.4f}）")

    # --- 4. 学習ループ（train2.pyと同一） ---
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': []
    }
    # 再開時はbest_f1を0からではなく引き継いだ値から始める。そうしないと、
    # 再開直後のまだ不安定なval F1でも「ベスト更新」と誤判定してしまう。
    best_f1 = RESUME_BEST_F1 if RESUME_FROM else 0.0
    early_stop_counter = 0
    all_preds, all_labels = [], []

    run_label = f"{model_name}, fold {fold}" if fold is not None else f"{model_name}, fixed split"
    print(f"[sliding_window] 学習開始 ({run_label}, Device: {device})")

    with open(os.path.join(save_dir, "training_log.txt"), "w", buffering=1) as f:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            print(f"Epoch {epoch + 1} 開始...")
            all_train_preds, all_train_labels = [], []

            optimizer.zero_grad()
            num_batches = len(train_loader)
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels) / grad_accum_steps
                loss.backward()

                is_last_batch = (i + 1) == num_batches
                if (i + 1) % grad_accum_steps == 0 or is_last_batch:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                    optimizer.step()
                    optimizer.zero_grad()

                running_loss += loss.item() * grad_accum_steps
                _, predicted = torch.max(outputs, 1)

                all_train_preds.extend(predicted.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())

                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                if (i + 1) % 50 == 0:
                    print(f"  Batch {i + 1}/{len(train_loader)} 処理中...")

            model.eval()
            all_preds, all_labels = [], []
            val_running_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()

                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    total_val += labels.size(0)
                    correct_val += (preds == labels).sum().item()

            history['train_loss'].append(running_loss / len(train_loader))
            history['val_loss'].append(val_running_loss / len(val_loader))
            history['train_acc'].append(correct_train / total_train)
            history['val_acc'].append(correct_val / total_val)
            train_f1 = f1_score(all_train_labels, all_train_preds, average='macro', zero_division=0)
            val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            history['train_f1'].append(train_f1)
            history['val_f1'].append(val_f1)

            log_msg = f"Epoch {epoch + 1}/{num_epochs} | Loss: {history['train_loss'][-1]:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}"
            print(log_msg)
            f.write(log_msg + "\n")

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                early_stop_counter = 0
                print("  [Best Model Saved]")
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print("Early stopping triggered!")
                    break

    # --- 5. 最終結果出力（train2.pyと同一） ---
    torch.save(model.state_dict(), os.path.join(save_dir, 'last_model.pth'))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_result.png'))
    plt.close()

    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    per_phase_f1 = f1_score(all_labels, all_preds, average=None, labels=range(num_classes), zero_division=0)
    phase_metrics = {}
    for i, phase_name in enumerate(active_phases):
        support = int(cm[i].sum())
        phase_acc = cm[i, i] / support if support > 0 else 0.0
        phase_metrics[phase_name] = {
            "accuracy": float(phase_acc),
            "f1": float(per_phase_f1[i]),
            "support": support,
        }
    with open(os.path.join(save_dir, "phase_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(phase_metrics, f, indent=2, ensure_ascii=False)

    print(f"[{run_label}] フェーズごとの Accuracy / F1:")
    with open(os.path.join(save_dir, "training_log.txt"), "a", encoding="utf-8") as f:
        f.write("\n--- フェーズごとの Accuracy / F1 ---\n")
        for phase_name, m in phase_metrics.items():
            line = f"  {phase_name}: Accuracy={m['accuracy']:.4f}, F1={m['f1']:.4f} (support={m['support']})"
            print(line)
            f.write(line + "\n")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=active_phases, yticklabels=active_phases
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    title_suffix = "" if INCLUDE_IDLE_PHASE else " without idle"
    fold_suffix = f" - {run_label}" if fold is not None else ""
    plt.title(f'Confusion Matrix - sliding_window ({num_classes} Classes{title_suffix}{fold_suffix})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_filename = 'confusion_matrix.png' if INCLUDE_IDLE_PHASE else 'confusion_matrix_no_idle.png'
    plt.savefig(os.path.join(save_dir, cm_filename))
    plt.close()

    print(f"[{run_label}] 学習終了。best val F1: {best_f1:.4f}（{save_dir} を確認してください）")

    del model, optimizer, criterion, train_loader, val_loader, train_dataset, val_dataset
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return best_f1


def main():
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} が見つかりません。先に build_sliding_window_index.py を実行してください。"
        )

    multi_model = len(MODEL_NAMES) > 1

    if USE_CROSS_VALIDATION:
        cv_root = Path("result") / f"epochs_{num_epochs}" / f"sliding_window_cv_{NUM_FOLDS}fold_{current_time}"
        os.makedirs(cv_root, exist_ok=True)

        all_summaries = {}
        for model_name in MODEL_NAMES:
            model_root = (cv_root / model_name) if multi_model else cv_root

            fold_f1s = []
            for fold in range(NUM_FOLDS):
                fold_dir = model_root / f"fold_{fold}"
                best_f1 = run_training(fold_dir, model_name=model_name, fold=fold)
                fold_f1s.append(best_f1)

            summary = {
                "model_name": model_name,
                "num_folds": NUM_FOLDS,
                "fold_best_val_f1": fold_f1s,
                "mean_val_f1": float(np.mean(fold_f1s)),
                "std_val_f1": float(np.std(fold_f1s)),
            }
            with open(model_root / "cv_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"[{model_name}] 交差検証終了。mean val F1: {summary['mean_val_f1']:.4f} ± {summary['std_val_f1']:.4f}")
            all_summaries[model_name] = summary

        if multi_model:
            with open(cv_root / "cv_summary_all_models.json", "w", encoding="utf-8") as f:
                json.dump(all_summaries, f, indent=2, ensure_ascii=False)
            print(f"全モデルの交差検証結果を {cv_root / 'cv_summary_all_models.json'} にまとめました。")
    else:
        for model_name in MODEL_NAMES:
            base_dir = Path("result") / f"epochs_{num_epochs}" / f"sliding_window_checkpoints_{current_time}"
            save_dir = (base_dir / model_name) if multi_model else base_dir
            run_training(save_dir, model_name=model_name, fold=None)


if __name__ == "__main__":
    main()
