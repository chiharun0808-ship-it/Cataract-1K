
import os
import subprocess
import math
import shutil
import numpy as np
from os.path import isfile, join
from PIL import Image
import random
import csv
from datetime import datetime, timedelta
#順番②
#メモ：フェーズごとに切り取った元動画を60fpsから30fpsに変換して新規作成
# パス設定
path1 = r"C:\Users\kit02\GitHub\Cataract-1K\Phase_recognition_dataset\phase_recognition_annotations/"

# フォルダ一覧を取得（caseで始まるフォルダのみ）
case_list = [d for d in os.listdir(path1) if d.startswith("case") and os.path.isdir(os.path.join(path1, d))]
case_list.sort()

# 各ケースごとの処理
for i in range(0, len(case_list)):
    case_folder = os.path.join(path1, case_list[i])

    subfolders = [f.path for f in os.scandir(case_folder) if f.is_dir()]

    for j in range(len(subfolders)):
        phase_folder = subfolders[j]
        vid_list = os.listdir(phase_folder)
        print(f'Processing folder: {phase_folder}')

        for k in range(len(vid_list)):
            filename = vid_list[k]

            # 【重要】動画ファイル（.mp4）のみを対象にし、30fps版は除外
            if not filename.endswith(".mp4") or "30fps" in filename:
                continue

            input_path = os.path.join(phase_folder, filename)
            output_filename = filename.replace(".mp4", "30fps.mp4")
            output_path = os.path.join(phase_folder, output_filename)

            # 既に変換済みならスキップ
            if os.path.exists(output_path):
                print(f"Skipping (already exists): {filename}")
                continue

            # FFmpegコマンド作成
            cmd = f'ffmpeg -y -i "{input_path}" -filter:v fps=30 "{output_path}"'

            # エラーハンドリング付きで実行
            try:
                print(f"Converting: {filename}...")
                subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
                print(f"Success: {output_filename}")
            except subprocess.CalledProcessError as e:
                # 失敗した場合はエラー内容を表示して次へ進む
                print(f"--- FAILED: {filename} ---")
                print(f"Error: {e.output.decode(errors='ignore')}")
                continue

print("すべての処理が完了しました。")