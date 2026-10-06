"""
案B の前処理。学習（lib/window_dataset.py）と推論（lib/streaming.py）の両方が、必ずこの関数を使う。

2026-10-05 ユーザー決定（計画の ②・切り出し a）:
初期値に使う torchvision の ViT-B/16（weights="DEFAULT" = IMAGENET1K_V1）の前提に、そのまま合わせる。
自分で書き写すと細部がずれるおそれがあるので、torchvision が重みと一緒に配布している前処理
（ViT_B_16_Weights.DEFAULT.transforms()）をそのまま呼ぶ。その中身は次のとおり（torchvision 0.17 で確認）:
  1. 短い辺を 256 に縮小（bilinear、antialias あり）  512×384 → 341×256
  2. 中央の 224×224 を切り出す（左右それぞれ約17%、上下それぞれ約6%が切れる）
  3. 0〜1 の float にする
  4. ImageNet の平均 (0.485, 0.456, 0.406)・標準偏差 (0.229, 0.224, 0.225) で正規化

入力は cv2 で読んだ BGR の uint8 配列（元動画のフレーム、または劣化なしで保存した PNG）。
"""
import cv2
import torch
from torchvision.models import ViT_B_16_Weights

_VIT_TRANSFORM = ViT_B_16_Weights.DEFAULT.transforms()

PREPROCESS_NAME = "torchvision_ViT_B_16_DEFAULT_transforms"


def preprocess_frame_vit(frame_bgr):
    """BGR uint8 (H, W, 3) → 正規化済み float テンソル (3, 224, 224)。"""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor_uint8 = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    return _VIT_TRANSFORM(tensor_uint8)


def describe():
    return repr(_VIT_TRANSFORM)
