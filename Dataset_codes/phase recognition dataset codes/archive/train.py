# [現在未使用・参考用] train2.py が正式版。
# ここでの import (`from prepare_dataset import ...` 等) は実際のフォルダ構成
# (dataset_index/, models/, dataset_weight/) と一致しておらず、このままでは動作しない。
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from prepare_dataset import SurgicalPhaseDataset
from cnn_rnn_hybrids import ResNet50LSTM
from class_weight import get_class_weights

# --- 1. 設定項目 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 4
learning_rate = 1e-4
num_epochs = 50
patience = 25
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

class_weights = get_class_weights().to(device)

# --- 2. データセットの準備 (重要: num_workers=0) ---
csv_path = "dataset_index_split.csv"
train_dataset = SurgicalPhaseDataset(csv_path, split='train')
val_dataset = SurgicalPhaseDataset(csv_path, split='val')

# Windows環境でのハングアップを防ぐため num_workers=0 を明示
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# --- 3. モデル・損失・最適化 ---
model = ResNet50LSTM(num_classes=13, pretrained_cnn=True).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# --- 4. 学習ループ ---
history = {'train_loss': [], 'val_f1': []}
best_f1 = 0.0
early_stop_counter = 0

print(f"学習開始 (Device: {device})")

with open(os.path.join(save_dir, "training_log.txt"), "w", buffering=1) as f:
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # 学習ループの進捗を可視化するため各バッチで print を出力
        print(f"Epoch {epoch + 1} 開始...")

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(f"  Batch {i + 1}/{len(train_loader)} 処理中...")

        # 検証
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images.to(device))
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro')
        epoch_loss = running_loss / len(train_loader)

        history['train_loss'].append(epoch_loss)
        history['val_f1'].append(val_f1)

        log_msg = f"Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Val F1: {val_f1:.4f}"
        print(log_msg)
        f.write(log_msg + "\n")

        # Early Stopping & Best保存
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
# グラフ描画
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_f1'], label='Val F1')
plt.legend()
plt.savefig(os.path.join(save_dir, 'training_result.png'))
print("学習終了。checkpointsディレクトリを確認してください。")