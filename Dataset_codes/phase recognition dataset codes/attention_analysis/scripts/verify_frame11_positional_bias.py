"""
analyze_temporal_attention.py で見つかった「11番目のフレームにAttentionが集中する」
現象が、(A) 内容に依存しない固定の位置バイアス(TimeSformerのtime_embed由来)なのか、
(B) たまたま11番目の位置に重要な映像内容が入りやすいだけなのか、を切り分ける検証スクリプト。

【検証方法（因果的な操作実験）】
実際のテストウィンドウから1枚の「本物の」フレーム画像を取り出し、それを30回複製して
1本の疑似ウィンドウ(30フレームすべてが完全に同一画像)を作る。

  - もし(A)位置バイアスが原因なら:
    30フレームの内容が完全に同一(=内容による差がゼロ)でも、TimeSformerのTemporal
    Attentionは特定の位置(11番目や30番目)に偏るはずである。なぜなら、Attentionへの
    入力には patch_embed(内容由来。今回は全フレーム同一なので寄与ゼロ)に加えて、
    time_embed(位置由来。models/timesformer.py の TimeSformer.forward:
    `x.reshape(B,T,S,D) + self.time_embed[:,:T,:].unsqueeze(2)` )が
    フレームごとに異なる値として加算されており、これは内容と無関係だから。

  - もし(B)内容駆動が原因なら:
    30フレームの内容を完全に同一にした時点で「11番目が特別」という理由が消えるため、
    Attentionはおおむね一様（または自己参照の30番目のみ強い）になるはずである。

この操作を、様々な症例・フェーズから抽出した複数の実フレームで繰り返し、結果が
毎回同じような位置的パターンを示すかを見る。特定の1枚だけでなく多数で再現すれば、
「その画像の内容」ではなく「位置そのもの」が原因であるという主張の裏付けが強くなる。

【安全性】
既存スクリプト・モデル・Datasetは一切変更しない。analyze_temporal_attention.py から
TemporalAttentionRecorder（読み取り専用のforward hook）と各種ローダーをそのまま
importして再利用する。

使い方:
    python verify_frame11_positional_bias.py [checkpoint_dir] [--n-probes 30] [--output-dir frame11_bias_check]
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

matplotlib.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from dataset_index.prepare_dataset import SurgicalPhaseDataset
from evaluation.test2 import load_eval_config, phases, phases_12
from analyze_temporal_attention import TemporalAttentionRecorder, build_model


def compute_frame_importance(recorder: TemporalAttentionRecorder, sequence_length: int) -> np.ndarray:
    """analyze_temporal_attention.py と全く同じ集約方法
    （query=最後のフレーム、head平均・layer平均。空間パッチはhook内で平均済み）。"""
    attn_stacked = recorder.stacked()  # (depth, heads, T, T)
    query_idx = sequence_length - 1
    return attn_stacked[:, :, query_idx, :].mean(dim=(0, 1)).numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Temporal Attentionの『11番目に集中』が位置バイアスか内容依存かを検証する"
    )
    parser.add_argument(
        "checkpoint_dir", nargs="?",
        default="result/epochs_35/sliding_window_checkpoints_0914_10_46",
    )
    parser.add_argument("--checkpoint-name", default="best_model.pth")
    parser.add_argument("--csv", default="data/dataset_index_sliding_window_stride5.csv")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--n-probes", type=int, default=30,
        help="何種類の異なる実フレームで『30枚とも同一画像』の疑似ウィンドウを作って試すか",
    )
    parser.add_argument("--output-dir", default="frame11_bias_check")
    parser.add_argument("--seed", type=int, default=0)
    # 実データでのframe_11の分散チェックに使う、既存のanalyze_temporal_attention.py出力
    parser.add_argument(
        "--real-attention-csv", default="attention_analysis/csv/attention_summary.csv",
        help="analyze_temporal_attention.py が既に生成したCSV（実内容での比較線に使う。無ければスキップ）",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_path = checkpoint_dir / args.checkpoint_name
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    dataset = SurgicalPhaseDataset(
        args.csv, split=args.split, sequence_length=sequence_length,
        frame_interval=1, exclude_idle=not include_idle_phase,
    )
    print(f"[verify] {args.split} split: {len(dataset)} ウィンドウから {args.n_probes} 種類の実フレームを抽出して検証")

    # 様々な症例・フェーズを広くカバーするため、データセット全体から均等間隔で
    # サンプリングする（先頭に偏らないようにするため。乱数ではなく等間隔サンプリングに
    # することで再現性も確保する）。
    n_probes = min(args.n_probes, len(dataset))
    probe_indices = np.linspace(0, len(dataset) - 1, n_probes, dtype=int)

    recorder = TemporalAttentionRecorder(model)

    synthetic_curves = []
    probe_rows = []

    with torch.no_grad():
        for probe_i, idx in enumerate(probe_indices):
            row_meta = dataset.data.iloc[int(idx)]
            case_id = row_meta["case_id"]
            true_label = row_meta["label"]

            clip_tensor, _ = dataset[int(idx)]  # (30, 3, 224, 224) 本物のウィンドウ
            # 30番目(最後、query/ラベル対象フレーム)の「本物の1枚」を種にする。
            seed_frame = clip_tensor[-1:]  # (1, 3, 224, 224)
            # これを30回複製 = 内容の時間変化を完全にゼロにした疑似ウィンドウ
            synthetic_clip = seed_frame.repeat(sequence_length, 1, 1, 1)  # (30, 3, 224, 224)

            recorder.per_layer.clear()
            _ = model(synthetic_clip.unsqueeze(0).to(device))
            curve = compute_frame_importance(recorder, sequence_length)
            synthetic_curves.append(curve)

            argmax_frame = int(np.argmax(curve)) + 1
            probe_rows.append({
                "probe_id": probe_i + 1,
                "source_window_case_id": case_id,
                "source_window_true_label": true_label,
                "argmax_frame_when_content_identical": argmax_frame,
                "argmax_value": float(np.max(curve)),
            })
            print(f"  probe {probe_i+1}/{n_probes}: 種フレーム={case_id}/{true_label} -> "
                  f"内容同一化後のargmax位置={argmax_frame}")

    recorder.remove()

    synthetic_curves = np.stack(synthetic_curves, axis=0)  # (n_probes, 30)
    mean_curve = synthetic_curves.mean(axis=0)
    std_curve = synthetic_curves.std(axis=0)

    probe_df = pd.DataFrame(probe_rows)
    for k in range(sequence_length):
        probe_df[f"frame_{k+1:02d}_attention"] = synthetic_curves[:, k]
    probe_csv = out_dir / "identical_content_probes.csv"
    probe_df.to_csv(probe_csv, index=False, encoding="utf-8-sig")

    # argmax位置の分布（内容を同一化した後でも11番目/30番目に偏るか）
    argmax_counts = probe_df["argmax_frame_when_content_identical"].value_counts().sort_index()

    # --- 比較用プロット: 内容同一化後(位置のみ) vs 実内容での平均カーブ ---
    real_mean_curve = None
    if Path(args.real_attention_csv).exists():
        real_df = pd.read_csv(args.real_attention_csv)
        frame_cols = [f"frame_{i:02d}_attention" for i in range(1, sequence_length + 1)]
        real_mean_curve = real_df[frame_cols].values.astype(float).mean(axis=0)

    x = np.arange(1, sequence_length + 1)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, mean_curve, "o-", color="#c8791e", label=f"内容を同一化(位置のみ, n={n_probes}probe平均)", zorder=3)
    ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color="#c8791e", alpha=0.15, zorder=2)
    if real_mean_curve is not None:
        ax.plot(x, real_mean_curve, "o-", color="#0b6e67",
                 label=f"実内容(analyze_temporal_attention.py, n={len(real_df)}件平均)", zorder=3)
    ax.axvline(11, color="#888888", linestyle=":", linewidth=1.2, zorder=1)
    ax.set_xlabel("ウィンドウ内フレーム番号 (1〜30、30=query/分類対象フレーム)")
    ax.set_ylabel("平均Temporal Attention重み")
    ax.set_title("内容を完全に同一化した場合 vs 実内容の場合の、Temporal Attention重要度カーブ比較")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "identical_content_vs_real_content.png", dpi=140)
    plt.close(fig)

    # --- 判定サマリーの出力 ---
    frac_at_11 = (probe_df["argmax_frame_when_content_identical"] == 11).mean()
    frac_at_30 = (probe_df["argmax_frame_when_content_identical"] == 30).mean()

    summary_lines = [
        "# frame11位置バイアス検証結果\n",
        f"検証プローブ数: {n_probes}（{args.split} splitから等間隔サンプリングした異なる実フレームを使用）\n",
        "",
        "## 内容を完全に同一化(30枚とも同じ画像)した場合のargmax位置の分布",
        argmax_counts.to_string(),
        "",
        f"- argmaxが11番目だった割合: {frac_at_11*100:.1f}%",
        f"- argmaxが30番目(自己)だった割合: {frac_at_30*100:.1f}%",
        "",
        "## 判定",
    ]
    if frac_at_11 >= 0.5:
        verdict = (
            "内容を完全に同一化しても、多くのプローブでargmaxが11番目に留まった。\n"
            "これは「11番目への集中」が映像内容ではなく、TimeSformerのtime_embed由来の\n"
            "固定的な位置バイアスであることを強く示唆する（内容の差がゼロでも再現するため）。"
        )
    elif frac_at_11 == 0.0:
        verdict = (
            "内容を完全に同一化すると、11番目へのargmax集中は全く再現しなかった。\n"
            "これは実データで見られた11番目への集中が、位置バイアスではなく、\n"
            "実際の映像内容（11番目付近に写っている何か）に依存していた可能性を示唆する。\n"
            "ただし、位置バイアスが『内容と組み合わさって初めて11番目を強めるように働く』\n"
            "可能性（純粋な位置バイアス単体では顕在化しないケース）は本実験だけでは排除できない。"
        )
    else:
        verdict = (
            "内容を同一化した場合でも一部のプローブで11番目にargmaxが残ったが、全てではない。\n"
            "位置バイアスの寄与はあるが、それだけで実データの集中度(37.2%)を完全には\n"
            "説明できない可能性がある。位置バイアスと内容の両方が寄与している可能性が高い。"
        )
    summary_lines.append(verdict)
    summary_lines.append("")
    summary_lines.append(
        "## 注意点\n"
        "- 種フレームには実際にテストウィンドウの最終フレーム(query位置の本物の画像)を使用。\n"
        "  ゼロ埋めなど不自然な画像は使っていない。\n"
        "- 30枚とも同一画像なので、patch_embedから来る内容由来の寄与はフレーム間で完全に\n"
        "  等しくなる。したがって位置ごとの差が残るなら、その差の発生源は\n"
        "  (a) time_embed（位置埋め込み、内容非依存）、または\n"
        "  (b) Attention計算そのものに内在する非対称性（数値的な理由等）\n"
        "  のいずれかであり、『特定フレームの映像内容』ではあり得ない。\n"
        "- 参考として実データ(analyze_temporal_attention.pyの881件平均)のカーブと重ねて\n"
        "  identical_content_vs_real_content.png に可視化した。"
    )

    readme_path = out_dir / "README.md"
    readme_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n=== 判定サマリー ===")
    print("\n".join(summary_lines))
    print(f"\n詳細: {probe_csv}")
    print(f"図: {out_dir / 'identical_content_vs_real_content.png'}")
    print(f"README: {readme_path}")


if __name__ == "__main__":
    main()
