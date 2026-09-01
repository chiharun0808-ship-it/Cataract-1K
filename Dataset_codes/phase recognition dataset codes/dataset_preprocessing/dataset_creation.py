

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
#順番④
#メモ：caseを横断して同じフェーズの30fpsの動画を集める
#path1 = 'phase_recognition_annotations/'
path1 = r"C:\Users\kit02\GitHub\Cataract-1K\Phase_recognition_dataset\phase_recognition_annotations/"
path2 = r"C:\Users\kit02\GitHub\Cataract-1K\Phase_recognition_dataset\Training_Dataset/"

#case_list = os.listdir(path1) #動画を読み込む
#synapse由来のcsvファイルがannotationsフォルダに含まれているのが原因でエラーがでたのでフォルダだけリストアップするように修正した。
case_list = [f for f in os.listdir(path1) if os.path.isdir(os.path.join(path1, f))]
case_list.sort()


for i in range(0,len(case_list)):
    case_folder = path1 + case_list[i]
    subfolders = [f.path for f in os.scandir(case_folder) if f.is_dir()]
    print(f'subfolders: {subfolders}')

    for j in range(len(subfolders)):

        files = os.listdir(subfolders[j])#caseフォルダ下のフェーズフォルダたち

        for k in range(len(files)):
            #os.makedirs(path2 + subfolders[j].split(os.sep)[-1], exist_ok = True)
            #shutil.copy(subfolders[j] + '/' + files[k], path2 + subfolders[j].split(os.sep)[-1] + '/' + files[k])
            filename = files[k]
            # 30fps.mp4 の文字列が含まれるファイルだけを対象にする
            if "30fps.mp4" in filename:
                # フェーズごとのフォルダを作成
                dest_dir = os.path.join(path2, os.path.basename(subfolders[j]))
                os.makedirs(dest_dir, exist_ok=True)
                # コピー実行
                src = os.path.join(subfolders[j], filename)
                dst = os.path.join(dest_dir, filename)
                shutil.copy(src, dst)