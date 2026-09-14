"""
正解フェーズ(アノテーションCSV)の読み込みと、動画fpsに応じたサンプリング間隔の
換算ロジック。visualize_phase_on_video.py / build_sliding_window_index.py /
realtime_phase_overlay.py など、切り出し前の動画を扱う複数スクリプトから共通で使う。
"""
import csv

# SurgicalPhaseDataset のデフォルト設定（frame_interval=10, 30fpsのクリップ前提）を
# 基準として、そのときの「サンプル間隔の実時間」を全動画共通の基準にする。
# 学習用クリップは全て30fpsで書き出されているが、切り出し前の手術動画はfpsが
# 異なる場合がある（例: 60fps）ため、fpsに応じてサンプル間隔（フレーム数）を
# 換算しないと、学習時と違う実時間スケールの映像を見せることになってしまう。
TRAIN_FPS_REFERENCE = 30
TRAIN_FRAME_INTERVAL_REFERENCE = 10  # 学習時: 10フレームごと(30fpsで1/3秒間隔)にサンプリング


def native_frame_interval(fps: float) -> int:
    """学習時の「1/3秒間隔」を、この動画のfpsでのフレーム数に換算する。"""
    return max(1, round(fps * TRAIN_FRAME_INTERVAL_REFERENCE / TRAIN_FPS_REFERENCE))


def normalize_phase_name(name: str) -> str:
    """アノテーションCSVとphasesリストで表記ゆれ（スペース/スラッシュ/大文字小文字）が
    あるため、比較用に正規化する（例: "Irrigation/Aspiration" と "Irrigation_Aspiration"）。"""
    return name.lower().replace(" ", "").replace("/", "").replace("_", "")


def load_ground_truth_intervals(gt_csv_path, active_phases):
    """
    アノテーションCSV（caseId, comment, frame, endFrame, sec, endSec）を読み込み、
    [(start_sec, end_sec, phase_name), ...] のリストを返す。phase_nameはactive_phasesの
    表記に正規化される。区間外の時刻は idle 扱いとする。
    """
    normalized_to_canonical = {normalize_phase_name(p): p for p in active_phases}
    intervals = []
    with open(gt_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = normalize_phase_name(row["comment"])
            canonical = normalized_to_canonical.get(key)
            if canonical is None:
                print(f"警告: アノテーションのフェーズ名 '{row['comment']}' がphasesリストと一致しません。無視します。")
                continue
            intervals.append((float(row["sec"]), float(row["endSec"]), canonical))
    return intervals


def true_phase_at(intervals, t_sec, idle_name="idle"):
    for start, end, name in intervals:
        if start <= t_sec <= end:
            return name
    return idle_name
