"""
手順8: 現行モデル（91.4%）と再調査のモデルを、同じ test 11症例・同じ時点で比べる。

入力:
- 現行モデル: 保存済みの予測ログ（attention_analysis/streaming_attention/pred_logs/。手順2で全時点の再現を確認済み）
- 再調査: <run-dir>/streaming_eval/pred_logs/
出力（<run-dir>/comparison/）:
- by_case.csv     症例ごとの一致率と、その差（新 − 現行）
- by_phase.csv    フェーズごとの recall と F1（13クラス）
- summary.json    全体の一致率、動画全体での macro F1、予測の切り替わり回数、差が正の症例数
- confusion_<model>.csv  混同行列（行=正解、列=予測）
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.paths import BASELINE_PRED_LOG_DIR, TEST_CASES

PHASES = [
    "Incision", "Viscoelastic", "Capsulorhexis", "Hydrodissection",
    "Phacoemulsification", "Irrigation_Aspiration", "CapsulePulishing",
    "LensImplantation", "LensPositioning", "Viscoelastic_Suction",
    "Anterior_ChamberFlushing", "Tonifying_Antibiotics", "idle",
]


def load(log_dir):
    return {c: pd.read_csv(Path(log_dir) / f"{c}.pred_log.csv") for c in TEST_CASES}


def switches(s):
    return int((s != s.shift()).iloc[1:].sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    out = run_dir / "comparison"
    out.mkdir(exist_ok=True)

    models = {"baseline_91.4": load(BASELINE_PRED_LOG_DIR), "defectfree": load(run_dir / "streaming_eval" / "pred_logs")}
    rows, summary, by_phase = [], {}, {}
    for name, logs in models.items():
        allp = pd.concat(logs.values())
        y, p = allp.true_phase, allp.pred_phase
        summary[name] = {
            "accuracy": round(float((y == p).mean() * 100), 2), "n_steps": len(allp),
            "macro_f1_video": round(float(f1_score(y, p, labels=PHASES, average="macro", zero_division=0)), 4),
            "pred_switches": sum(switches(d.pred_phase) for d in logs.values()),
            "true_switches": sum(switches(d.true_phase) for d in logs.values()),
        }
        by_phase[f"{name}_recall"] = recall_score(y, p, labels=PHASES, average=None, zero_division=0)
        by_phase[f"{name}_f1"] = f1_score(y, p, labels=PHASES, average=None, zero_division=0)
        pd.DataFrame(confusion_matrix(y, p, labels=PHASES), index=PHASES, columns=PHASES).to_csv(out / f"confusion_{name}.csv")
    for c in TEST_CASES:
        a = models["baseline_91.4"][c]; b = models["defectfree"][c]
        if len(a) != len(b) or not (a.true_phase.values == b.true_phase.values).all():
            raise SystemExit(f"[停止] {c}: 2つの予測ログで時点や正解が一致しません")
        acc_a = (a.true_phase == a.pred_phase).mean() * 100
        acc_b = (b.true_phase == b.pred_phase).mean() * 100
        rows.append({"case_id": c, "n_steps": len(a), "baseline_acc": round(acc_a, 2),
                     "defectfree_acc": round(acc_b, 2), "diff": round(acc_b - acc_a, 2)})
    by_case = pd.DataFrame(rows)
    by_case.to_csv(out / "by_case.csv", index=False)
    pd.DataFrame(by_phase, index=PHASES).round(4).to_csv(out / "by_phase.csv")
    summary["cases_improved"] = int((by_case["diff"] > 0).sum())
    summary["cases_worse"] = int((by_case["diff"] < 0).sum())
    summary["mean_case_diff"] = round(float(by_case["diff"].mean()), 2)
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(by_case.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
