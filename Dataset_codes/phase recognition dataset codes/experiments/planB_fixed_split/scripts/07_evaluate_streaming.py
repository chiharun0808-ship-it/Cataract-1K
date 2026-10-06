"""
手順7: test 11症例のストリーミング評価（91.4% と同じ手順。前処理だけ案B のもの）。

- lib/streaming.py の iter_stream_clips で、元動画を先頭から読み、1秒ごとに30枚の窓を作る
- モデルの出力の最大のクラスを予測とする（平滑化なし。既存の評価の smooth_window=1 と同じ）
- 正解は窓の末尾の時刻の正解フェーズ（utils/gt_utils.true_phase_at）
- 症例ごとに pred_logs/<case>.pred_log.csv（time_sec, pred_phase, true_phase）を書く。既存と同じ形式

評価スクリプト自体が正しいかを確かめるため、--legacy を付けると「現行モデル＋旧来の前処理
（inference/visualize_phase_on_video.py の preprocess_frame）」で動かせる。この場合の予測ログは、
保存済みの予測ログ（91.4%）と全時点で一致するはず。

実行例（phase recognition dataset codes/ 直下で）:
    python experiments/planB_fixed_split/scripts/07_evaluate_streaming.py --run-dir <runs/...>
    python experiments/planB_fixed_split/scripts/07_evaluate_streaming.py --legacy --cases case_4697 --out-dir <...>
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.paths import BASELINE_CHECKPOINT_DIR, CODES_DIR, GT_CSV_TEMPLATE, TEST_CASES, VIDEO_TEMPLATE
from lib.preprocess import preprocess_frame_vit
from lib.streaming import iter_stream_clips

sys.path.insert(0, str(CODES_DIR))
from inference.visualize_phase_on_video import preprocess_frame as preprocess_frame_legacy  # noqa: E402
from models.timesformer import create_timesformer_model  # noqa: E402
from utils.gt_utils import load_ground_truth_intervals, true_phase_at  # noqa: E402

PHASES = [
    "Incision", "Viscoelastic", "Capsulorhexis", "Hydrodissection",
    "Phacoemulsification", "Irrigation_Aspiration", "CapsulePulishing",
    "LensImplantation", "LensPositioning", "Viscoelastic_Suction",
    "Anterior_ChamberFlushing", "Tonifying_Antibiotics", "idle",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", help="学習の run フォルダ（best_model.pth を使う）。出力は <run-dir>/streaming_eval/")
    parser.add_argument("--checkpoint", default="best_model.pth")
    parser.add_argument("--legacy", action="store_true", help="現行モデル＋旧来の前処理で動かす（評価スクリプトの確認用）")
    parser.add_argument("--cases", nargs="*", default=TEST_CASES)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.legacy:
        ckpt = BASELINE_CHECKPOINT_DIR / "best_model.pth"
        preprocess = preprocess_frame_legacy
    else:
        ckpt = Path(args.run_dir) / args.checkpoint
        preprocess = preprocess_frame_vit
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.run_dir) / "streaming_eval"
    (out_dir / "pred_logs").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_timesformer_model("timesformer", num_classes=len(PHASES), num_frames=30, pretrained_cnn=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"チェックポイント: {ckpt}\n前処理: {'旧来（antialias なし・正規化なし）' if args.legacy else '案B（事前学習と同じ）'}")

    total = correct = 0
    for case in args.cases:
        t0 = time.time()
        gt = load_ground_truth_intervals(GT_CSV_TEMPLATE.format(case=case), PHASES)
        rows = []
        with torch.no_grad():
            for _, t_sec, clip in iter_stream_clips(VIDEO_TEMPLATE.format(case=case), preprocess, stride_sec=1.0):
                pred = PHASES[int(model(clip.unsqueeze(0).to(device)).argmax(1))]
                rows.append((t_sec, pred, true_phase_at(gt, t_sec)))
        df = pd.DataFrame(rows, columns=["time_sec", "pred_phase", "true_phase"])
        df.to_csv(out_dir / "pred_logs" / f"{case}.pred_log.csv", index=False)
        acc = (df.pred_phase == df.true_phase).mean()
        total += len(df); correct += int((df.pred_phase == df.true_phase).sum())
        print(f"  {case}: {len(df)} 時点、一致率 {acc * 100:.2f}%（{time.time() - t0:.0f} 秒）", flush=True)
    print(f"全体の一致率: {correct / total * 100:.2f}%（{correct}/{total}）")


if __name__ == "__main__":
    main()
