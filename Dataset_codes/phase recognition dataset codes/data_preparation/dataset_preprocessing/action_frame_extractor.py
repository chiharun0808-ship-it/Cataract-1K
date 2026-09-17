import os
import subprocess
import csv
from datetime import timedelta
#順番①
#メモ：caseファイル下にフェーズごとのファイルを作成し、該当部を切り取ったmp4ファイルを入れる
# 設定：パスはご自身の環境（Phase_recognition_datasetなど）に合わせてください
path = r"/Phase_recognition_dataset/phase_recognition_annotations/"
video_path = r"/Phase_recognition_dataset/dataset_videos/phase_recognition/"

case_list = [d for d in os.listdir(path) if d.startswith("case") and os.path.isdir(os.path.join(path, d))]#caseで始まるフォルダのみ取得
case_list.sort()#並び替え


def csv_read_annot(csv_file):
    action_inf = []
    action_name = []
    with open(csv_file, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            action_name.append(row[1])#comment列(2列目)からフェーズ名を読み込み
            action_inf.append([float(row[2]), float(row[3])])#開始フレームと終了フレーム数を読み込み
    return action_inf, action_name


def csv_read_fps(csv_file):
    with open(csv_file, 'r') as file:
        csvreader = csv.reader(file)
        next(csvreader)
        for row in csvreader:
            return float(row[4]) #csvファイルのfps列(5列目)から読み込み(元動画は60fps)


def frame_to_secs(in_frame, fps) -> float:
    return int(in_frame) / fps #フレーム数から秒数に変換


def secs_to_timedelta(sec_float: float):
    return timedelta(seconds=int(sec_float), microseconds=(sec_float * 1000000) % 1000000)


def convert_frames_to_video(vid, start, duration, output_path):
    # ffmpegの引数をダブルクォーテーションで囲み、スペースや特殊文字に対応
    cmd = f'ffmpeg -i "{vid}" -ss {start} -t {duration} -c:v copy -c:a copy "{output_path}"'
    subprocess.check_output(cmd, shell=True)


for i in range(0, len(case_list)):
    case_folder = os.path.join(path, case_list[i])
    video_inf_csv = os.path.join(case_folder, f"{case_list[i]}_video.csv")
    video_annot_csv = os.path.join(case_folder, f"{case_list[i]}_annotations_phases.csv")
    video = os.path.abspath(os.path.join(video_path, f"{case_list[i]}.mp4"))

    fps = csv_read_fps(video_inf_csv)#csvファイルからfps数を取得
    annot, name = csv_read_annot(video_annot_csv)#annotに開始終了フレーム、nameにフェーズ名が入る

    for j in range(len(annot)):#1動画に含まれるフェーズの数だけ繰り返し
        # Irrigation/Aspiration 等のスラッシュやスペースを安全な名前に置換
        clean_name = name[j].replace(" ", "").replace("/", "_")#csvファイル内でのフェーズ名は空白やスラッシュがあるから扱いやすいように変換する
        action_folder = os.path.join(case_folder, clean_name)
        os.makedirs(action_folder, exist_ok=True)#case別フォルダの下にフェーズごとにフォルダを作成

        frame_start, frame_stop = annot[j]#フェーズごとの開始終了フレーム
        if (frame_stop - frame_start) / fps > 3:#3秒以上のフェーズについて
            start_secs = frame_to_secs(frame_start, fps)
            duration_secs = frame_to_secs(frame_stop, fps) - start_secs

            start_str = str(secs_to_timedelta(start_secs))
            duration_str = str(secs_to_timedelta(duration_secs))

            # ファイル名からもコロンを取り除く（Windows用）
            file_name = f"{case_list[i]}_{clean_name}_{start_str.replace(':', '_')}-{str(secs_to_timedelta(frame_to_secs(frame_stop, fps))).replace(':', '_')}.mp4"
            output_path = os.path.join(action_folder, file_name)

            convert_frames_to_video(video, start_str, duration_str, output_path)#算出したそのフェーズの開始終了秒数に基づいて動画から該当部を切り取る。ファイル名にも秒数が含まれている。