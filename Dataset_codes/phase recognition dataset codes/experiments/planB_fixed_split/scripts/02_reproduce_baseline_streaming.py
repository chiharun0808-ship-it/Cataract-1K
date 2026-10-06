"""
手順2: 現行モデルの 91.4% を、同じ評価コードで再現できるか確かめる。

- 既存の評価スクリプト（attention_analysis/scripts/streaming_attention_eval.py）を、
  91.4% を出したときと同じ引数（--no-attention, stride 1秒, 平滑化なし, test 11症例）で実行する。
  スクリプトとチェックポイントは元のチェックアウトにあるものを読むだけで、変更しない。
- 出力は runs/<日付>_baseline-reproduce_current-checkpoint/ に書く。
- 保存済みの予測ログ（attention_analysis/streaming_attention/pred_logs/）と、全時点で
  time_sec・pred_phase・true_phase が一致するかを確かめる。1つでも違えば終了コード1。

実行（どこからでもよい）:
    python experiments/planB_fixed_split/scripts/02_reproduce_baseline_streaming.py
    python experiments/planB_fixed_split/scripts/02_reproduce_baseline_streaming.py --skip-inference  # 照合だけやり直す
"""
import argparse
import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.paths import (BASELINE_CHECKPOINT_DIR, BASELINE_PRED_LOG_DIR, BASELINE_STREAMING_SCRIPT,
                       RUNS_DIR, SOURCE_CODES_DIR, TEST_CASES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None, help="出力先（省略時は runs/<今日>_baseline-reproduce_current-checkpoint）")
    parser.add_argument("--skip-inference", action="store_true", help="推論はせず、既にある出力の照合だけ行う")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else RUNS_DIR / f"{date.today().isoformat()}_baseline-reproduce_current-checkpoint"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(BASELINE_STREAMING_SCRIPT), str(BASELINE_CHECKPOINT_DIR.relative_to(SOURCE_CODES_DIR)),
           "--no-attention", "--stride-sec", "1.0", "--smooth-window", "1",
           "--cases", *TEST_CASES, "--output-dir", str(run_dir / "streaming_eval")]
    if not args.skip_inference:
        print("実行:", " ".join(cmd))
        print(f"（作業ディレクトリ: {SOURCE_CODES_DIR}。既存のファイルは書き換えない）")
        with open(run_dir / "inference.log", "w", encoding="utf-8") as log:
            subprocess.run(cmd, cwd=SOURCE_CODES_DIR, check=True, stdout=log, stderr=subprocess.STDOUT)

    # 照合
    rows, problems = [], []
    total = correct_new = correct_old = 0
    for case in TEST_CASES:
        old = pd.read_csv(BASELINE_PRED_LOG_DIR / f"{case}.pred_log.csv")
        new = pd.read_csv(run_dir / "streaming_eval" / "pred_logs" / f"{case}.pred_log.csv")
        same_len = len(old) == len(new)
        if not same_len:
            problems.append(f"{case}: 時点数が違う（保存済み {len(old)}、再実行 {len(new)}）")
            n_diff = None
        else:
            diff = (old["time_sec"].round(3) != new["time_sec"].round(3)) | \
                   (old["pred_phase"] != new["pred_phase"]) | (old["true_phase"] != new["true_phase"])
            n_diff = int(diff.sum())
            if n_diff:
                first = diff.idxmax()
                problems.append(f"{case}: {n_diff} 時点で不一致（最初は time_sec={old.loc[first, 'time_sec']}: "
                                f"保存済み pred={old.loc[first, 'pred_phase']}, 再実行 pred={new.loc[first, 'pred_phase']}）")
        acc_old = (old.pred_phase == old.true_phase).mean()
        acc_new = (new.pred_phase == new.true_phase).mean()
        total += len(new); correct_new += int((new.pred_phase == new.true_phase).sum())
        correct_old += int((old.pred_phase == old.true_phase).sum())
        rows.append({"case_id": case, "steps_saved": len(old), "steps_rerun": len(new),
                     "acc_saved": round(acc_old * 100, 2), "acc_rerun": round(acc_new * 100, 2),
                     "n_mismatch": n_diff})

    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "reproduction_check.csv", index=False, encoding="utf-8")
    overall_new = correct_new / total * 100
    print(table.to_string(index=False))
    print(f"\n全体の一致率: 再実行 {overall_new:.2f}% （{correct_new}/{total}）")

    meta = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(BASELINE_CHECKPOINT_DIR / "best_model.pth"),
        "command": cmd, "cwd": str(SOURCE_CODES_DIR),
        "overall_accuracy_rerun": round(overall_new, 4),
        "total_steps": total,
        "reproduced_exactly": not problems,
        "problems": problems,
    }
    with open(run_dir / "reproduction_check.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if problems:
        print("\n[不一致] 保存済みの予測ログと一致しませんでした:")
        for p in problems:
            print("  - " + p)
        sys.exit(1)
    print("[OK] 全時点で保存済みの予測ログと完全に一致しました。")


if __name__ == "__main__":
    main()
