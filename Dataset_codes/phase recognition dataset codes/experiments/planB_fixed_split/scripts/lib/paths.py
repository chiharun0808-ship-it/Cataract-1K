"""
案B のスクリプトが共通で使うパス。

このフォルダ（experiments/planB_fixed_split）は git の作業用ツリーにも置かれる。
作業用ツリーには git の管理対象外のもの（元動画、既存の予測ログ、既存のチェックポイント、
mp4v の学習用クリップ）が無いので、それらは元のチェックアウト（SOURCE_CODES_DIR）から
**読むだけ**で使う。新しい出力はすべて PLAN_DIR の中に書く。
"""
from pathlib import Path

PLAN_DIR = Path(__file__).resolve().parents[2]       # experiments/planB_fixed_split
CODES_DIR = PLAN_DIR.parents[1]                      # このチェックアウトの phase recognition dataset codes

# 元のチェックアウト（git 管理外のデータがある場所）。読むだけで、書き込まない。
SOURCE_CODES_DIR = Path(r"C:\Users\kit02\GitHub\Cataract-1K\Dataset_codes\phase recognition dataset codes")
DATASET_DIR = SOURCE_CODES_DIR.parents[1] / "Phase_recognition_dataset"

VIDEO_TEMPLATE = str(DATASET_DIR / "dataset_videos" / "phase_recognition" / "{case}.mp4")
GT_CSV_TEMPLATE = str(DATASET_DIR / "phase_recognition_annotations" / "{case}" / "{case}_annotations_phases.csv")

# 既存の結果（比較の基準。読むだけ）
BASELINE_CHECKPOINT_DIR = SOURCE_CODES_DIR / "result" / "epochs_35" / "sliding_window_checkpoints_0914_10_46"
BASELINE_PRED_LOG_DIR = SOURCE_CODES_DIR / "attention_analysis" / "streaming_attention" / "pred_logs"
BASELINE_STREAMING_SCRIPT = SOURCE_CODES_DIR / "attention_analysis" / "scripts" / "streaming_attention_eval.py"
BASELINE_WINDOW_CSV = SOURCE_CODES_DIR / "data" / "dataset_index_sliding_window_stride5.csv"

# 案B の出力
CASE_SPLIT_CSV = PLAN_DIR / "data" / "case_split_56cases.csv"
RUNS_DIR = PLAN_DIR / "runs"
FRAMES_DIR = PLAN_DIR / "frames"
CONDITIONS_DIR = PLAN_DIR / "conditions"

TEST_CASES = ["case_4697", "case_4709", "case_4811", "case_4852", "case_4874", "case_4924",
              "case_5015", "case_5063", "case_5300", "case_5315", "case_5331"]
