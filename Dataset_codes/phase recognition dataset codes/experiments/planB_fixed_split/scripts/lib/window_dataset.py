"""
案B の学習データの読み込み。

窓の一覧（data/windows_repro-baseline_f30_i333ms_s5s.csv）の1行が1サンプル。
窓の30枚のフレームを PNG（元動画のフレームを劣化なしで保存したもの）から読み、
推論（lib/streaming.py）と**同じ前処理関数**（lib/preprocess.py の preprocess_frame_vit）をかける。
このため、学習と推論で入力の作り方が一致する（確認は scripts/05_check_train_inference_input_match.py）。

参考にした既存ファイル（変更していない）:
- dataset_index/prepare_dataset.py の SurgicalPhaseDataset（返す形 (30, 3, 224, 224) とラベル）
"""
from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from .paths import FRAMES_DIR
from .preprocess import preprocess_frame_vit

# 画像は「元動画のフレームそのもの」なので、前処理を変えても取り出し直す必要はない
DEFAULT_FRAME_ROOT = FRAMES_DIR / "repro-baseline"


class WindowFrameDataset(Dataset):
    def __init__(self, window_csv, split, frame_root=DEFAULT_FRAME_ROOT, preprocess=preprocess_frame_vit):
        df = pd.read_csv(window_csv)
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.frame_root = Path(frame_root)
        self.preprocess = preprocess
        print(f"[{split}] {len(self.data)} 窓")

    def __len__(self):
        return len(self.data)

    def frame_path(self, case_id, idx):
        return self.frame_root / case_id / f"{idx:06d}.png"

    def __getitem__(self, i):
        row = self.data.iloc[i]
        frames = []
        for idx in map(int, row["frame_indices"].split()):
            bgr = cv2.imread(str(self.frame_path(row["case_id"], idx)), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(self.frame_path(row["case_id"], idx))
            frames.append(self.preprocess(bgr))
        return torch.stack(frames), torch.tensor(int(row["label_id"]), dtype=torch.long)
