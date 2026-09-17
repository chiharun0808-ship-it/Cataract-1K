"""
TimeSformer (Divided Space-Time Attention) の Temporal Attention 解析スクリプト。

【目的】
30フレームの因果的スライディングウィンドウ（末尾フレーム時点のフェーズを分類する）に対して、
TimeSformerが分類時に「どの時間位置(フレーム)の情報を強く参照しているか」を、
Temporal Attentionの重みから可視化・記録する。特に、フェーズ切り替えを含むウィンドウで
切り替え付近に注目しているかどうかを確認することが主目的。

【このスクリプトが変更しないもの（安全性）】
- models/timesformer.py, dataset_index/prepare_dataset.py, train2.py, train3.py,
  test2.py, visualize_phase_on_video.py, realtime_phase_overlay.py, gt_utils.py,
  build_sliding_window_index.py は一切変更しない。すべてimportして「読み取り専用」で使う。
- モデルの重み・forward計算そのものも変更しない。Attention重みの取得は
  forward hook（読み取り専用のコールバック）でのみ行う。

────────────────────────────────────────────────────────
【① 既存コードの整理（このスクリプトの前提）】
────────────────────────────────────────────────────────
- 入力: build_sliding_window_index.py が生成した dataset_index_sliding_window_stride5.csv。
  各行は「実時間1/3秒間隔で既にサンプリング済みの30フレームクリップ」で、ラベルは
  ウィンドウ末尾フレーム時点のフェーズ(causal)。
- Dataset: dataset_index/prepare_dataset.py の SurgicalPhaseDataset を
  frame_interval=1 で使う（クリップは既に30枚ちょうどなので二重サンプリングしない。
  train3.py と同じ設定）。
- モデル: models/timesformer.py の TimeSformer。checkpoint は train2.py/train3.py が
  保存した best_model.pth + config.json（test2.py の load_eval_config と同じ読み方）。
- 13クラスの定義・順序は test2.py の phases / phases_12 をそのまま使う。
- Ground Truthは gt_utils.py の load_ground_truth_intervals / true_phase_at を
  そのまま使う（visualize_phase_on_video.py と同一ロジック）。
- 症例ごとのGT CSVパスは build_sliding_window_index.py の GT_CSV_TEMPLATE を
  そのまま使う（新しいパス規則を勝手に作らない）。

────────────────────────────────────────────────────────
【④ Temporal Attentionの取得方法（このファイルの核心部分）】
────────────────────────────────────────────────────────
models/timesformer.py の _Attention.forward() は、softmax後のattention重み自体を
外部に返さない（returnされるのはprojection後の出力のみ）。モデル本体を書き換えずに
これを取得するため、TimeSformerBlock.temporal_attn（_Attentionインスタンス）に
forward hookを登録し、hookの中で _Attention.forward と全く同じ手順
（qkv線形層 → reshape → softmax）を「読み取り専用」で再計算する。
_Attention は num_heads / head_dim / scale / qkv をすべて公開属性・サブモジュールとして
持つため、モデルのforward計算自体には一切影響を与えずに正確な値を再現できる。
（実装: TemporalAttentionRecorder クラルを参照）

Tensor shapeの流れ（1ウィンドウ=1サンプルとして処理する。B=1固定。理由は下記参照）:
  temporal_attn への入力 x: (N=B*S, L=T, D) = (1*196, 30, 768)
    B=1（1ウィンドウずつ処理）, S=196（14x14空間パッチ）, T=30（フレーム数）, D=768（埋め込み次元）
  qkv(x) → reshape → (3, N, heads=12, T=30, head_dim=64) → q, k
  attn = softmax((q @ k^T) * scale)  → shape (N=196, heads=12, T=30 query, T=30 key)

【なぜB=1固定か】
上のattnはN(=B*S)次元を持つ。もしB>1で複数ウィンドウを同時にバッチ処理すると、
N次元には「複数ウィンドウの空間パッチ」が混在してしまい、後述の「空間パッチ方向に平均」
という集約を行った際に、異なるウィンドウの情報が混ざってしまう。これを避けるため、
本スクリプトは常に1ウィンドウ=1バッチで推論する。

────────────────────────────────────────────────────────
【フレーム単位の重要度への集約方法（曖昧さを排除するため明記）】
────────────────────────────────────────────────────────
Temporal Attentionは (layer, head, spatial patch, query frame, key frame) という
5つの次元を持つ。30フレーム単位の重要度1本に集約するにあたり、各次元の扱いを以下の通り
明確に固定する:

  1. query frame（どのフレームについて「何を参照したか」を見るか）:
     → 固定で「最後のフレーム(30枚目, 0-indexedで29)」を使う。
       このウィンドウのラベルは、まさにこの最後のフレーム時点のフェーズであるため、
       「分類対象フレームが、他のどのフレームの情報を時間Attentionで取り込んでいるか」
       を直接見ることが、本解析の目的（切り替え位置を参照しているか）に最も直結する。
       他のquery frameについては平均を取らない（平均すると「最後のフレームの判断根拠」
       という焦点がぼやけるため）。

  2. key frame（＝出力される30長のベクトルの軸そのもの。集約しない）
     → これがそのままX軸「フレーム番号1〜30」になる。

  3. spatial patch方向: 平均する。
     → Temporal Attentionは空間位置ごとに独立に計算されているため、画像全体として
       「どの時刻を平均的に参照しているか」を見るために平均する。hook内で
       attn.mean(dim=0)（N=B*S, B=1なのでS方向の平均）として計算時点で縮約する
       （保存データ量を抑えるため。空間解像度ごとの詳細を残したい場合は将来の拡張課題）。

  4. head方向: 平均する。
     → 各headが異なるパターンを学習している可能性はあるが、まずは単一の「重要度カーブ」
       として解釈しやすくするため、12ヘッドの単純平均を取る。

  5. layer(block)方向: 平均する。
     → 12層分の (heads平均後の) attention行を単純平均して最終的な30長のベクトルにする。
       ただし「層ごとに参照パターンが変わる」可能性を握りつぶさないよう、層平均する前の
       生データ (depth=12, heads=12, query=30, key=30; spatial patchは既に平均済み) を
       1ウィンドウごとに .npy として別途保存する（csv/plots同様、後から層別に
       再分析できるようにするため）。

  結果として、最終的な「30フレームの重要度」は:
    frame_importance[key] = mean over layers, mean over heads of
                             attn[layer, head, query=29, key]  for key in 0..29
  という定義になる。各値はsoftmaxの出力（確率）を平均したものなので、おおむね
  合計が1に近い値になる（=最後のフレームが持つ「参照の総量」を30フレームに配分した値）。

────────────────────────────────────────────────────────
【重要：Attentionの解釈について】
────────────────────────────────────────────────────────
「Attentionが高い＝そのフレームが分類の直接的な原因」と断定することはできない。
Attentionは中間表現同士の類似度に基づく重み付けであり、モデルの最終判断（clsトークン経由）
に対する厳密な因果的説明ではない（Attentionの解釈可能性については機械学習コミュニティでも
議論がある）。本解析はあくまで「モデルが時間方向のどこを平均的に参照する傾向があるか」を
定量化するものとして扱うこと。この注意書きはCSV出力・生成されるREADME・各プロットの
フッターにも明記する。

────────────────────────────────────────────────────────
【Ground Truthのフェーズ境界判定】
────────────────────────────────────────────────────────
sliding-window CSVには各ウィンドウの「末尾フレーム時点」のsource_time_secしか
記録されていない（30フレーム個別のタイムスタンプは保存されていない）。
build_sliding_window_index.py の設計上、ウィンドウ内の30フレームは実時間で
ちょうど1/3秒間隔（gt_utils.TRAIN_FRAME_INTERVAL_REFERENCE / TRAIN_FPS_REFERENCE）に
なるようサンプリングされているため、本スクリプトでは
    frame_time[k] = max(0, source_time_sec - (29 - k) * (1/3))   (k = 0..29, 0-indexed)
としてフレームごとの時刻を「再構成」する（動画冒頭でバッファが埋まっていない区間の
パディングは、時刻0にクランプすることで近似している。真の値と数ms〜数十ms程度
ずれる可能性があるが、境界検出の目的には十分な精度）。
再構成した各時刻について gt_utils.true_phase_at() でフェーズを引き、30フレーム分の
フェーズ列を作る。列の中に2種類以上のフェーズが含まれていれば「フェーズ切り替えを含む
ウィンドウ」と判定し、切り替え直後の最初のフレーム番号(1-indexed)を transition_frame
として記録する。全30フレームが同一フェーズなら transition_frame は空(NaN)。
なお、正解/不正解の判定や true_label 自体は、この再構成値ではなく
sliding-window CSVのlabel列（build_sliding_window_index.py が実フレーム番号から
厳密に計算した値）をそのまま使う。フレーム再構成はあくまで「ウィンドウ内の
どこで切り替わったか」という可視化目的にのみ使用する。

使い方:
    python analyze_temporal_attention.py [checkpoint_dir] [--csv CSV] [--split test]
        [--output-dir attention_analysis] [--max-samples 5] [--only-errors]

    例（まず5件だけ動作確認）:
    python analyze_temporal_attention.py \\
        result/epochs_35/sliding_window_checkpoints_0914_10_46 \\
        --max-samples 5

    例（誤分類ウィンドウだけ、最大50件）:
    python analyze_temporal_attention.py \\
        result/epochs_35/sliding_window_checkpoints_0914_10_46 \\
        --only-errors --max-samples 50
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# プロット上の日本語ラベルが文字化け(tofu表示)しないよう、Windows環境で利用可能な
# 日本語対応フォントを明示的に指定する（このスクリプト内のみの設定。既存スクリプトの
# matplotlib設定には影響しない）。
matplotlib.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
import pandas as pd
import torch

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from dataset_index.prepare_dataset import SurgicalPhaseDataset
from models.timesformer import create_timesformer_model, TIMESFORMER_MODELS
from evaluation.test2 import load_eval_config, phases, phases_12
from utils.gt_utils import (
    TRAIN_FPS_REFERENCE,
    TRAIN_FRAME_INTERVAL_REFERENCE,
    load_ground_truth_intervals,
    true_phase_at,
)
from data_preparation.build_sliding_window_index import GT_CSV_TEMPLATE, VIDEO_DIR_TEMPLATE

FRAME_SPACING_SEC = TRAIN_FRAME_INTERVAL_REFERENCE / TRAIN_FPS_REFERENCE  # = 1/3秒

ATTENTION_CAVEAT = (
    "注意: Attentionが高い = そのフレームが分類の直接的な原因、とは限らない。"
    "ここでは「モデルが時間方向のどこを平均的に参照する傾向があるか」を示す。"
)


class TemporalAttentionRecorder:
    """
    TimeSformerの各ブロックの temporal_attn (_Attentionインスタンス) にforward hookを
    登録し、softmax後のTemporal Attention重みを「読み取り専用」で記録するクラス。

    models/timesformer.py の _Attention.forward() の内部計算をhook内で再現するだけで、
    モデル本体のforward()やパラメータには一切変更を加えない。

    register()後、モデルに1ウィンドウ分（B=1）を forward すると、self.per_layer に
    層ごとの (heads, T, T) テンソル（空間パッチ方向は平均済み、CPU上のtensor）が貯まる。
    """

    def __init__(self, model: torch.nn.Module):
        self.per_layer = {}
        self._handles = []
        for layer_idx, block in enumerate(model.blocks):
            handle = block.temporal_attn.register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            # module = block.temporal_attn（_Attentionインスタンス）
            # inputs[0] = norm_temporal適用後の入力 x, shape (N=B*S, L=T, D)
            # これは models/timesformer.py の _Attention.forward() 冒頭と全く同じ入力。
            x = inputs[0]
            N, L, D = x.shape
            # 以下3行は _Attention.forward() の該当部分（qkv計算〜softmax）の再現。
            # モジュール自体の qkv線形層・num_heads・head_dim・scale をそのまま使うので、
            # 実際の計算と完全に一致する（読み取り専用、モデルへの書き込みなし）。
            qkv = module.qkv(x).reshape(N, L, 3, module.num_heads, module.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, N, heads, L, head_dim)
            q, k = qkv[0], qkv[1]
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)  # (N=B*S, heads, T, T)

            # 空間パッチ方向(N=B*S。B=1固定なのでN=Sそのもの)に平均。
            # 集約方針の詳細はファイル冒頭のdocstring参照。
            attn_spatial_mean = attn.mean(dim=0)  # (heads, T, T)
            self.per_layer[layer_idx] = attn_spatial_mean.detach().to("cpu")

        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def stacked(self) -> torch.Tensor:
        """層0..depth-1の順に積んだ (depth, heads, T, T) テンソルを返す。"""
        depth = len(self.per_layer)
        return torch.stack([self.per_layer[i] for i in range(depth)], dim=0)


def reconstruct_frame_times(source_time_sec: float, sequence_length: int = 30) -> list:
    """ウィンドウ内30フレームの実時間タイムスタンプを再構成する（ファイル冒頭docstring参照）。
    末尾フレーム(index=sequence_length-1)がsource_time_secになるよう、1/3秒間隔で
    過去にさかのぼる。動画冒頭のパディング区間は時刻0にクランプして近似する。"""
    return [
        max(0.0, source_time_sec - (sequence_length - 1 - k) * FRAME_SPACING_SEC)
        for k in range(sequence_length)
    ]


def detect_transition(frame_phases: list):
    """frame_phases(長さ30のフェーズ名リスト)の中で、最初にフェーズが切り替わる
    フレーム番号(1-indexed)を返す。切り替えが無ければ (False, None, 0) を返す。
    複数回切り替わる場合も n_transitions で件数を記録する。"""
    transition_indices_0idx = [
        i for i in range(1, len(frame_phases)) if frame_phases[i] != frame_phases[i - 1]
    ]
    if not transition_indices_0idx:
        return False, None, 0
    first_transition_frame_1idx = transition_indices_0idx[0] + 1  # 切り替え後最初のフレーム
    return True, first_transition_frame_1idx, len(transition_indices_0idx)


def build_model(model_name: str, num_classes: int, sequence_length: int, device: torch.device):
    if model_name not in TIMESFORMER_MODELS:
        raise ValueError(
            f"analyze_temporal_attention.py はTimeSformer専用です（Divided Space-Time "
            f"AttentionのTemporal Attentionを前提に実装しているため）。model_name={model_name} "
            f"は未対応です。"
        )
    return create_timesformer_model(
        model_name, num_classes=num_classes, num_frames=sequence_length, pretrained_cnn=False
    ).to(device)


def plot_window(
    out_path: Path,
    frame_importance: np.ndarray,
    true_label: str,
    pred_label: str,
    correct: bool,
    confidence: float,
    has_transition: bool,
    transition_frame,
    case_id: str,
    window_id: str,
    time_range_sec: tuple,
):
    n_frames = len(frame_importance)
    x = np.arange(1, n_frames + 1)
    query_frame = n_frames  # 1-indexedで30番目 = 最後のフレーム = ラベル対象フレーム

    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = ["#c8791e" if i == query_frame else "#0b6e67" for i in x]
    ax.bar(x, frame_importance, color=colors, width=0.75, zorder=3)
    ax.set_xlabel("ウィンドウ内フレーム番号 (1〜30、30=分類対象フレーム)")
    ax.set_ylabel("平均Temporal Attention重み\n(query=最後のフレーム, head/layer/空間平均)")
    ax.set_xticks(list(range(1, n_frames + 1, 2)) + [n_frames])
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

    if has_transition and transition_frame is not None:
        ax.axvline(
            transition_frame - 0.5, color="#b23b2f", linestyle="--", linewidth=1.6, zorder=4,
            label=f"GTフェーズ切り替え(直前:{transition_frame-1}→直後:{transition_frame})",
        )
        ax.legend(loc="upper left", fontsize=9)

    result_word = "正解" if correct else "誤分類"
    result_color = "#0b6e67" if correct else "#b23b2f"
    title = (
        f"{window_id}  |  {case_id}  |  t={time_range_sec[0]:.1f}-{time_range_sec[1]:.1f}s\n"
        f"正解: {true_label}   予測: {pred_label} ({confidence*100:.1f}%)   "
    )
    ax.set_title(title, fontsize=11, loc="left")
    ax.text(
        0.99, 1.02, result_word, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=11, fontweight="bold", color=result_color,
    )
    fig.text(0.01, 0.01, ATTENTION_CAVEAT, fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="TimeSformerのTemporal Attentionを解析し、30フレームの時間重要度を可視化・記録する"
    )
    parser.add_argument(
        "checkpoint_dir", nargs="?",
        default="result/epochs_35/sliding_window_checkpoints_0914_10_46",
        help="train3.py が保存した result/.../checkpoints_.../ ディレクトリ（省略時は最新のsliding-window学習結果）",
    )
    parser.add_argument("--checkpoint-name", default="best_model.pth")
    parser.add_argument(
        "--csv", default="data/dataset_index_sliding_window_stride5.csv",
        help="build_sliding_window_index.py が生成したインデックスCSV",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default="attention_analysis")
    parser.add_argument(
        "--max-samples", type=int, default=5,
        help="解析するウィンドウ数の上限（--only-errors指定時はこの件数の誤分類ウィンドウを集めるまで走査する）",
    )
    parser.add_argument(
        "--only-errors", action="store_true",
        help="誤分類ウィンドウのみを解析対象にする（指定しない場合は遭遇順にmax-samples件を解析）",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_path = checkpoint_dir / args.checkpoint_name
    out_dir = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    raw_dir = out_dir / "raw"
    csv_dir = out_dir / "csv"
    for d in (plots_dir, raw_dir, csv_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- ②③ 既存のconfig読み込み・モデル構築ロジックをそのまま再利用 ---
    eval_config = load_eval_config(checkpoint_dir)
    model_name = eval_config["model_name"]
    include_idle_phase = eval_config["include_idle_phase"]
    num_classes = eval_config["num_classes"]
    sequence_length = eval_config["sequence_length"]
    active_phases = phases if include_idle_phase else phases_12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_classes, sequence_length, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # --- ① sliding-window CSV(SurgicalPhaseDataset経由)から対象split・frame_interval=1で読む ---
    # frame_interval=1: build_sliding_window_index.py が既に30枚サンプリング済みのクリップを
    # 書き出しているため、ここで間引くと二重サンプリングになってしまう（train3.pyと同じ注意）。
    dataset = SurgicalPhaseDataset(
        args.csv, split=args.split, sequence_length=sequence_length,
        frame_interval=1, exclude_idle=not include_idle_phase,
    )
    print(f"[analyze] {args.split} split: {len(dataset)} ウィンドウ (checkpoint: {checkpoint_path})")

    recorder = TemporalAttentionRecorder(model)

    gt_intervals_cache = {}

    def get_gt_intervals(case_id: str):
        if case_id not in gt_intervals_cache:
            gt_csv_path = GT_CSV_TEMPLATE.format(case=case_id)
            gt_intervals_cache[case_id] = load_ground_truth_intervals(gt_csv_path, active_phases)
        return gt_intervals_cache[case_id]

    rows = []
    n_analyzed = 0
    n_scanned = 0

    with torch.no_grad():
        for idx in range(len(dataset)):
            if n_analyzed >= args.max_samples:
                break
            n_scanned += 1

            # SurgicalPhaseDataset.data は公開属性（pandas DataFrame）なので、
            # dataset[idx] と同じ行のメタデータをここから直接取得できる。
            # Dataset自体は一切変更していない（既存の__getitem__をそのまま呼ぶだけ）。
            row_meta = dataset.data.iloc[idx]
            case_id = row_meta["case_id"]
            source_time_sec = float(row_meta["source_time_sec"])
            file_path = row_meta["file_path"]

            # B=1固定でforward（複数ウィンドウを同時にバッチ処理しない理由はdocstring参照）
            clip_tensor, label_tensor = dataset[idx]
            true_label_id = int(label_tensor.item())
            true_label = active_phases[true_label_id]

            recorder.per_layer.clear()
            logits = model(clip_tensor.unsqueeze(0).to(device))  # (1, num_classes)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
            pred_label_id = int(probs.argmax().item())
            pred_label = active_phases[pred_label_id]
            confidence = float(probs[pred_label_id].item())
            correct = pred_label_id == true_label_id

            if args.only_errors and correct:
                continue

            # --- GTフェーズ境界の判定（可視化目的。docstring参照） ---
            frame_times = reconstruct_frame_times(source_time_sec, sequence_length)
            gt_intervals = get_gt_intervals(case_id)
            frame_phases = [true_phase_at(gt_intervals, t) for t in frame_times]
            has_transition, transition_frame, n_transitions = detect_transition(frame_phases)

            # --- Temporal Attentionの集約（docstring「フレーム単位の重要度への集約方法」参照） ---
            attn_stacked = recorder.stacked()  # (depth, heads, T, T) 空間パッチ方向は既に平均済み
            query_idx = sequence_length - 1  # 最後のフレーム = ラベル対象フレーム
            # layer方向・head方向に平均 → (T,) の30長ベクトル
            frame_importance = attn_stacked[:, :, query_idx, :].mean(dim=(0, 1)).numpy()

            window_id = f"window_{n_analyzed + 1:04d}"
            plot_path = plots_dir / f"{window_id}.png"
            raw_path = raw_dir / f"{window_id}_attn.npy"

            start_t = max(0.0, source_time_sec - (sequence_length - 1) * FRAME_SPACING_SEC)
            plot_window(
                plot_path, frame_importance, true_label, pred_label, correct, confidence,
                has_transition, transition_frame, case_id, window_id, (start_t, source_time_sec),
            )
            # 層別・head平均後(空間パッチも平均済み)の生データを保存。層方向に平均する前の
            # (depth=12, T=30 query, T=30 key) を残しておく（head方向のみ平均、layerは残す）。
            np.save(raw_path, attn_stacked.mean(dim=1).numpy())  # (depth, T_query, T_key)

            row = {
                "window_id": window_id,
                "case_id": case_id,
                "video_id": case_id,
                "source_case_video": VIDEO_DIR_TEMPLATE.format(case=case_id),
                "window_clip_path": file_path,
                "window_start_sec": round(start_t, 3),
                "window_end_sec": round(source_time_sec, 3),
                "true_label": true_label,
                "true_label_id": true_label_id,
                "predicted_label": pred_label,
                "predicted_label_id": pred_label_id,
                "correct": correct,
                "confidence": round(confidence, 4),
                "has_transition": has_transition,
                "transition_frame": transition_frame if transition_frame is not None else "",
                "n_transitions": n_transitions,
                "max_attention_frame": int(np.argmax(frame_importance)) + 1,
                "max_attention_value": round(float(np.max(frame_importance)), 5),
                "plot_path": str(plot_path).replace("\\", "/"),
                "raw_attn_path": str(raw_path).replace("\\", "/"),
            }
            for k in range(sequence_length):
                row[f"frame_{k+1:02d}_attention"] = round(float(frame_importance[k]), 5)
            rows.append(row)

            n_analyzed += 1
            status = "誤分類" if not correct else "正解"
            trans_note = f" transition@{transition_frame}" if has_transition else ""
            print(
                f"  [{n_analyzed}/{args.max_samples}] {window_id} {case_id} t={source_time_sec:.1f}s "
                f"true={true_label} pred={pred_label} ({status}){trans_note}"
            )

    recorder.remove()

    summary_csv = csv_dir / "attention_summary.csv"
    df = pd.DataFrame(rows)
    df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    readme_path = out_dir / "README.md"
    readme_path.write_text(_build_readme(args, len(dataset), n_scanned, n_analyzed, checkpoint_path), encoding="utf-8")

    print(f"\n完了。{n_analyzed}件のウィンドウを解析しました（{n_scanned}件走査）。")
    print(f"  サマリーCSV: {summary_csv}")
    print(f"  プロット: {plots_dir}/")
    print(f"  生Attentionデータ: {raw_dir}/")
    print(f"  説明: {readme_path}")


def _build_readme(args, dataset_size, n_scanned, n_analyzed, checkpoint_path) -> str:
    return f"""# Temporal Attention 解析結果

このディレクトリは `analyze_temporal_attention.py` の出力です。
このスクリプトは既存の学習・評価・推論スクリプト（train2.py, train3.py, test2.py,
visualize_phase_on_video.py, realtime_phase_overlay.py, models/timesformer.py,
dataset_index/prepare_dataset.py 等）を一切変更していません。すべて読み取り専用で
import して利用しています（Attention重みの取得も forward hook による読み取り専用の
再計算で、モデルのforward計算自体は変更していません）。

## 実行条件
- checkpoint: `{checkpoint_path}`
- インデックスCSV: `{args.csv}` (split={args.split})
- 対象split内の総ウィンドウ数: {dataset_size}
- 走査したウィンドウ数: {n_scanned} / 解析（保存）したウィンドウ数: {n_analyzed}
- only_errors: {args.only_errors}

## ディレクトリ構成
```
{args.output_dir}/
├── plots/window_XXXX.png   1ウィンドウ=1枚。30フレームのTemporal Attention重要度の棒グラフ。
│                           GTフェーズ切り替えがウィンドウ内にある場合のみ境界線を表示。
├── raw/window_XXXX_attn.npy  層別(depth=12)・head平均後・空間パッチ平均後の生データ。
│                             shape=(12, 30, 30) = (layer, query_frame, key_frame)。
│                             CSVのframe_XX_attention列は、これをさらにlayer方向に
│                             平均し、query=30番目(最後のフレーム)の行を取り出したもの。
└── csv/attention_summary.csv  1行=1ウィンドウ。列の意味は下記。
```

## attention_summary.csv の列
- window_id, case_id, video_id, source_case_video, window_clip_path
- window_start_sec, window_end_sec: ウィンドウの実時間範囲(再構成値。下記参照)
- true_label, true_label_id: 正解ラベル(sliding-window CSVのlabel列そのもの。厳密値)
- predicted_label, predicted_label_id, correct, confidence: モデルの予測
- has_transition, transition_frame, n_transitions: GTフェーズ切り替えの有無・位置(1-indexed)・回数
  ※ transition_frameは「切り替え後、最初に新フェーズになるフレーム番号」
  ※ 30フレーム個別の時刻はCSVに保存されていないため、ウィンドウ末尾時刻から
     1/3秒間隔で逆算した近似値を使って判定している（詳細は analyze_temporal_attention.py
     冒頭のdocstring「Ground Truthのフェーズ境界判定」参照）
- max_attention_frame, max_attention_value: 30フレーム中でAttentionが最大だったフレームと値
- plot_path, raw_attn_path: 対応するプロット・生データのパス
- frame_01_attention 〜 frame_30_attention: フレームごとのTemporal Attention重要度
  （query=最後のフレーム、head平均・layer平均・空間パッチ平均後の値。集約方法の詳細は
  analyze_temporal_attention.py 冒頭のdocstringを参照）

## Attentionの解釈について（重要）
{ATTENTION_CAVEAT}
Attentionは中間表現同士の類似度に基づく重み付けであり、モデルの最終判断に対する
厳密な因果的説明ではない。ここでの結果は「モデルが時間方向のどこを平均的に参照する
傾向があるか」を示す記述的な指標として解釈すること。

## 研究上の見方の例
- has_transition=Trueのウィンドウで、max_attention_frameがtransition_frame付近に
  集中しているか（＝切り替わり周辺の情報を参照する傾向があるか）を、
  正解ウィンドウと誤分類ウィンドウで比較する。
- correct=Falseのウィンドウで、Attentionのピークが「間違えた予測先のフェーズが
  支配的な時間帯」に寄っていないか（＝古いフェーズの情報に引きずられて
  誤分類している可能性）を確認する。
- raw/*.npy を使えば、層(layer)ごとにAttentionパターンが変化するか
  （浅い層は自己参照が強く、深い層ほど遠いフレームを参照する、等の傾向）も分析できる。
"""


if __name__ == "__main__":
    main()
