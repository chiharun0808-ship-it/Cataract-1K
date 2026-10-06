"""
案B（planB_fixed_split）用の症例分割ファイルを作る。

## 目的
既存の固定分割（54症例: train 32 / val 11 / test 11）を1症例も変えずに引き継ぎ、
前処理のバグで抜けていた case_5353・case_5357 を train として追加した
56症例の分割（train 34 / val 11 / test 11）を作る。

## 方針（2026-10-02 にユーザーと合意）
- 既存54症例の split は data/dataset_index_split.csv から「写す」だけで、分け直さない。
  dataset_index/create_index.py は乱数で分け直すため使わない（使うと test が入れ替わる）。
- 追加2症例は train に入れる。val・test は既存の11症例ずつのまま。
- 既存のファイルは読むだけで、変更・上書きしない。

## 確認すること（1つでも失敗したら出力を書かずに終了コード1で止まる）
1. 既存の分割CSV（クリップ単位）で、各症例の split が1つに決まっていること。
2. 既存のスライディングウィンドウ用CSVの split とも全症例で一致すること。
3. 配布データ（アノテーションのフォルダと元動画）が56症例そろっていること。
4. 既存の分割にない症例が、追加予定の2症例とちょうど一致すること。
5. 書き出したファイルを読み直し、既存54症例の split が元と完全に一致すること。
6. 実行の前後で、既存の分割CSVの SHA-256 が変わっていないこと。

## 実行方法（phase recognition dataset codes/ 直下で）
    python experiments/planB_fixed_split/scripts/01_build_case_split.py
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# --- パス（このファイルの位置から決める。カレントディレクトリに依存しない） ---
PLAN_DIR = Path(__file__).resolve().parents[1]          # experiments/planB_fixed_split
CODES_DIR = PLAN_DIR.parents[1]                         # phase recognition dataset codes
DATASET_DIR = CODES_DIR.parents[1] / "Phase_recognition_dataset"

SOURCE_SPLIT_CSV = CODES_DIR / "data" / "dataset_index_split.csv"
SOURCE_WINDOW_CSV = CODES_DIR / "data" / "dataset_index_sliding_window_stride5.csv"
ANNOTATION_DIR = DATASET_DIR / "phase_recognition_annotations"
VIDEO_DIR = DATASET_DIR / "dataset_videos" / "phase_recognition"

OUTPUT_CSV = PLAN_DIR / "data" / "case_split_56cases.csv"
OUTPUT_META = PLAN_DIR / "data" / "case_split_56cases.meta.json"

ADDED_CASES = ["case_5353", "case_5357"]
ADDED_SPLIT = "train"
ADDED_REASON = ("前処理スクリプトのループ範囲のバグ（len(case_list)-4）で学習データから"
                "抜けていた症例。2026-10-02 の合意により、既存の分割を変えずに train に追加")
EXPECTED_COUNTS = {"train": 34, "val": 11, "test": 11}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def case_split_map(csv_path: Path) -> dict:
    """クリップ単位のCSVから case_id -> split を作る。1症例に複数の split があれば失敗。"""
    df = pd.read_csv(csv_path)
    result, problems = {}, []
    for case_id, group in df.groupby("case_id"):
        splits = sorted(group["split"].unique())
        if len(splits) != 1:
            problems.append(f"{case_id}: split が複数あります {splits}")
        else:
            result[case_id] = splits[0]
    return result, problems


def fail(messages):
    print("\n[停止] 確認に失敗したため、出力ファイルは書き出していません。")
    for m in messages:
        print("  - " + m)
    sys.exit(1)


def main():
    hash_before = {p.name: sha256(p) for p in (SOURCE_SPLIT_CSV, SOURCE_WINDOW_CSV)}

    # 確認1: 既存の分割CSVで各症例の split が1つに決まっている
    split_map, problems = case_split_map(SOURCE_SPLIT_CSV)
    if problems:
        fail(problems)
    print(f"[確認1] 既存の分割CSV: {len(split_map)} 症例、各症例の split は1つ → OK")

    # 確認2: スライディングウィンドウ用CSVの split と一致する
    window_map, problems = case_split_map(SOURCE_WINDOW_CSV)
    if problems:
        fail(problems)
    mismatches = []
    for case_id in sorted(set(split_map) | set(window_map)):
        a, b = split_map.get(case_id), window_map.get(case_id)
        if a != b:
            mismatches.append(f"{case_id}: dataset_index_split.csv={a}, "
                              f"dataset_index_sliding_window_stride5.csv={b}")
    if mismatches:
        fail(mismatches)
    print("[確認2] スライディングウィンドウ用CSVとも全症例で split が一致 → OK")

    # 確認3: 配布データが56症例そろっている
    ann_cases = sorted(d.name for d in ANNOTATION_DIR.iterdir()
                       if d.is_dir() and d.name.startswith("case"))
    video_cases = sorted(p.stem for p in VIDEO_DIR.glob("case_*.mp4"))
    if ann_cases != video_cases or len(ann_cases) != 56:
        fail([f"アノテーション {len(ann_cases)} 症例、元動画 {len(video_cases)} 症例で一致しません",
              f"アノテーションのみ: {sorted(set(ann_cases) - set(video_cases))}",
              f"元動画のみ: {sorted(set(video_cases) - set(ann_cases))}"])
    print("[確認3] アノテーションと元動画が56症例そろっている → OK")

    # 確認4: 既存の分割にない症例 = 追加予定の2症例
    missing = sorted(set(ann_cases) - set(split_map))
    extra = sorted(set(split_map) - set(ann_cases))
    if missing != sorted(ADDED_CASES) or extra:
        fail([f"既存の分割にない症例: {missing}（想定: {ADDED_CASES}）",
              f"配布データにない症例が既存の分割にある: {extra}"])
    print(f"[確認4] 既存の分割にない症例は {missing} で想定どおり → OK")

    # 分割ファイルを組み立てる（既存54症例は写すだけ）
    rows = []
    for case_id in ann_cases:
        if case_id in split_map:
            rows.append({"case_id": case_id, "split": split_map[case_id],
                         "source": "existing_split",
                         "note": "data/dataset_index_split.csv の split をそのまま引き継ぎ"})
        else:
            rows.append({"case_id": case_id, "split": ADDED_SPLIT,
                         "source": "added_planB", "note": ADDED_REASON})
    out = pd.DataFrame(rows, columns=["case_id", "split", "source", "note"])

    counts = out["split"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        fail([f"split ごとの症例数が想定と違います: {counts}（想定: {EXPECTED_COUNTS}）"])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    # 確認5: 書き出したファイルを読み直して、既存54症例が元と完全に一致する
    reloaded = pd.read_csv(OUTPUT_CSV)
    reloaded_map = dict(zip(reloaded["case_id"], reloaded["split"]))
    mismatches = [f"{c}: 元={s}, 新={reloaded_map.get(c)}（原因: 書き出し後の値が元と異なる）"
                  for c, s in split_map.items() if reloaded_map.get(c) != s]
    if mismatches:
        OUTPUT_CSV.unlink()
        fail(mismatches)
    print(f"[確認5] 書き出したファイルで既存 {len(split_map)} 症例の split が完全に一致 → OK")

    # 確認6: 既存のCSVが変わっていない
    hash_after = {p.name: sha256(p) for p in (SOURCE_SPLIT_CSV, SOURCE_WINDOW_CSV)}
    if hash_before != hash_after:
        fail([f"既存ファイルの SHA-256 が変わりました: {hash_before} -> {hash_after}"])
    print("[確認6] 既存の分割CSVは変更されていない（SHA-256 が実行前後で同じ） → OK")

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "experiments/planB_fixed_split/scripts/01_build_case_split.py",
        "purpose": "案B: 既存の固定分割を保ったまま、抜けていた2症例を train に追加した56症例の分割",
        "source_files": {
            "data/dataset_index_split.csv": hash_before[SOURCE_SPLIT_CSV.name],
            "data/dataset_index_sliding_window_stride5.csv": hash_before[SOURCE_WINDOW_CSV.name],
        },
        "added_cases": {c: ADDED_SPLIT for c in ADDED_CASES},
        "added_reason": ADDED_REASON,
        "counts": counts,
        "cases_by_split": {s: sorted(out.loc[out["split"] == s, "case_id"]) for s in ["train", "val", "test"]},
        "not_used": "既存の fold 列（旧モデルの交差検証用）は案B では使わないため引き継いでいない",
        "checks_passed": [1, 2, 3, 4, 5, 6],
    }
    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n完了: {OUTPUT_CSV.relative_to(CODES_DIR)} を作成しました。")
    print(f"      {OUTPUT_META.relative_to(CODES_DIR)} に作成の記録を保存しました。")
    print(f"split ごとの症例数: {counts}")


if __name__ == "__main__":
    main()
