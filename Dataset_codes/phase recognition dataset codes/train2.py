import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
from dataset_index.prepare_dataset import SurgicalPhaseDataset
from models.cnn_rnn_hybrids import ResNet50LSTM
from dataset_weight.class_weight import get_class_weights
from pathlib import Path
#from class_weight_without_idle import get_class_weights
from datetime import datetime

#from create_index import phases

phases = [
    "Incision", "Viscoelastic", "Capsulorhexis", "Hydrodissection",
    "Phacoemulsification", "Irrigation_Aspiration", "CapsulePulishing",
    "LensImplantation", "LensPositioning", "Viscoelastic_Suction",
    "Anterior_ChamberFlushing", "Tonifying_Antibiotics", "idle"
]

#idleフェーズを分類から削除して実験したい。元に戻すときは以下をコメントアウトして、上のphase_13をphasesに直したらおっけい
phases_12 = [
    "Incision", "Viscoelastic", "Capsulorhexis", "Hydrodissection",
    "Phacoemulsification", "Irrigation_Aspiration", "CapsulePulishing",
    "LensImplantation", "LensPositioning", "Viscoelastic_Suction",
    "Anterior_ChamberFlushing", "Tonifying_Antibiotics"
]

# --- 1. 設定項目 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 4
learning_rate = 1e-4
num_epochs = 20
patience = 50
#save_dir = "checkpoints_epoch100"
# 現在の日時を取得してフォーマット (例: 7月21日0時49分 -> 0721_0_49)
current_time = datetime.now().strftime("%m%d_%H_%M")
#save_dir = f"checkpoints_epoch100_{current_time}"
# 【pathlibを使用】resultディレクトリの下に checkpoints_epoch100_{current_time} を指定
#save_dir = Path("result") / f"checkpoints_epoch{num_epochs}_{current_time}"
# 【変更】result の下にエポック数ごとのディレクトリを挟み、その中にタイムスタンプフォルダを作成する
save_dir = Path("result") / f"epochs_{num_epochs}" / f"checkpoints_{current_time}"

os.makedirs(save_dir, exist_ok=True)

class_weights = get_class_weights().to(device)

# --- 2. データセットの準備 ---
csv_path = "dataset_index_split.csv"
train_dataset = SurgicalPhaseDataset(csv_path, split='train')
val_dataset = SurgicalPhaseDataset(csv_path, split='val')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# --- 3. モデル・損失・最適化 ---
#idleを除外して実験するために13から12に変更
model = ResNet50LSTM(num_classes=13, pretrained_cnn=True).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- 4. 学習ループ ---
# 精度や損失を記録してグラフ描画に使用する辞書
history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': [],
    'train_f1': [], 'val_f1': []
}
best_f1 = 0.0
early_stop_counter = 0

print(f"学習開始 (Device: { device})")

with open(os.path.join(save_dir, "training_log.txt"), "w", buffering=1) as f:
    for epoch in range(num_epochs):
        # --- 学習フェーズ ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        print(f"Epoch {epoch + 1} 開始...")
        all_train_preds, all_train_labels = [], [] #追加した

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            # 予測値と正解ラベルを蓄積
            all_train_preds.extend(predicted.cpu().numpy())  # ← 追加
            all_train_labels.extend(labels.cpu().numpy())  # ← 追加

            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if (i + 1) % 50 == 0:
                print(f"  Batch {i + 1}/{len(train_loader)} 処理中...")

        # --- 検証フェーズ ---
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

        # --- 履歴への追加とログ記録 ---
        history['train_loss'].append(running_loss / len(train_loader))
        history['val_loss'].append(val_running_loss / len(val_loader))
        history['train_acc'].append(correct_train / total_train)
        history['val_acc'].append(correct_val / total_val)
        #val_f1 = f1_score(all_labels, all_preds, average='macro')
        # 訓練データと検証データの両方でF1値を算出
        train_f1 = f1_score(all_train_labels, all_train_preds, average='macro', zero_division=0)  # ← 追加
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        #log_msg = f"Epoch {epoch + 1}/{num_epochs} | Loss: {history['train_loss'][-1]:.4f} | Val F1: {val_f1:.4f}"
        history['train_f1'].append(train_f1)  # ← 追加
        history['val_f1'].append(val_f1)

        log_msg = f"Epoch {epoch + 1}/{num_epochs} | Loss: {history['train_loss'][-1]:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}"  # ← Train F1 を表示に追加
        print(log_msg)
        f.write(log_msg + "\n")

        # Early Stopping & Bestモデル保存
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

# --- 5. 最終結果出力 ---
torch.save(model.state_dict(), os.path.join(save_dir, 'last_model.pth'))

# LossとAccuracyのグラフを生成・保存
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

# 混合行列を生成・保存 以下は13フェーズ分出力されなかった
#cm = confusion_matrix(all_labels, all_preds)
#plt.figure(figsize=(10, 8))
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#plt.title('Confusion Matrix')
#plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
#そのため全部表示されるように修正
# 13クラスすべて（0から12）を強制的に含めるように labels を指定
cm = confusion_matrix(all_labels, all_preds, labels=range(13)) #idleフェーズ除外のために12に変更

plt.figure(figsize=(12, 10)) # クラス数が増えるため、少し図のサイズを大きくすると見やすくなります
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=phases, # 必要に応じてフェーズ名を入れると見やすくなります
    yticklabels=phases
)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (13 Classes)')
#plt.title('Confusion Matrix (12 Classes without idle)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
#plt.savefig(os.path.join(save_dir, 'confusion_matrix_no_idle.png'))
plt.close()


print("学習終了。checkpointsディレクトリを確認してください。")