import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
from torchvision import transforms


class SurgicalPhaseDataset(Dataset):
    """
    手術動画のフェーズ認識用データセットクラス
    CSVからsplit（train/val/test）に応じてデータをフィルタリングし、
    指定したシーケンス長とフレーム間隔でフレームを抽出・パディングします。
    """

    def __init__(self, csv_file, split='train', fold=None, frame_interval=10, sequence_length=30, transform=None, exclude_idle=False):
        # CSVを読み込み、指定された split ('train', 'val', 'test') で抽出
        full_df = pd.read_csv(csv_file)

        if fold is None or split == 'test':
            # 固定split（分割固定）モード: split列 ('train'/'val'/'test') でそのまま抽出
            self.data = full_df[full_df['split'] == split].reset_index(drop=True)
        elif split == 'val':
            # 交差検証モード: 指定foldに属する症例をvalとする（test症例は fold=-1 のため含まれない）
            self.data = full_df[full_df['fold'] == fold].reset_index(drop=True)
        elif split == 'train':
            # 交差検証モード: 指定fold以外に属する症例をtrainとする（test症例は含めない）
            self.data = full_df[(full_df['fold'] != fold) & (full_df['fold'] != -1)].reset_index(drop=True)
        else:
            raise ValueError(f"交差検証モードでは split は 'train' か 'val' を指定してください: {split}")

        # idleフェーズ(label_id=12)を除外するかどうかを呼び出し側から明示的に指定する
        if exclude_idle:
            self.data = self.data[self.data['label_id'] != 12].reset_index(drop=True)

        self.frame_interval = frame_interval
        self.sequence_length = sequence_length


        # モデル入力用のリサイズ設定
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
        ])

        fold_note = f" (fold={fold})" if fold is not None and split != 'test' else ""
        print(f"[{split}{fold_note}] データセット初期化: {len(self.data)} クリップ")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = self.data.iloc[idx]['file_path']
        label = self.data.iloc[idx]['label_id']

        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_idx = 0

        # 指定したフレーム間隔で動画からフレームを抽出
        while len(frames) < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.frame_interval == 0:
                # BGR -> RGB変換
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Tensorへの変換: (H, W, C) -> (C, H, W)
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

                # リサイズ処理
                frame_tensor = self.transform(frame_tensor)

                frames.append(frame_tensor)
            frame_idx += 1
        cap.release()

        # パディング処理（フレーム数が足りない場合の補完）
        if len(frames) == 0:
            # 動画が読み込めない異常ケース：ゼロ埋めテンソルを作成
            frames = [torch.zeros((3, 224, 224))] * self.sequence_length
        elif len(frames) < self.sequence_length:
            # 足りない分を最後のフレームで埋める
            last_frame = frames[-1]
            while len(frames) < self.sequence_length:
                frames.append(last_frame)

        # モデル入力用スタック: (Sequence, Channel, Height, Width)
        # 形状: [30, 3, 224, 224]
        return torch.stack(frames), torch.tensor(label, dtype=torch.long)


# --- 動作確認用 ---
if __name__ == "__main__":
    # 正しくsplit指定ができるか確認
    dataset = SurgicalPhaseDataset("../dataset_index_split.csv", split='train', frame_interval=10, sequence_length=30)

    if len(dataset) > 0:
        video_tensor, label = dataset[0]
        print(f"Sample 0 | Shape: {video_tensor.shape} | Label: {label}")
    else:
        print("データがありません。split名を確認してください。")