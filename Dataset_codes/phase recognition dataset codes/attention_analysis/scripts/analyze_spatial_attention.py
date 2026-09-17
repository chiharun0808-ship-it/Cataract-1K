"""
TimeSformer (Divided Space-Time Attention) の Spatial Attention 解析スクリプト。

analyze_temporal_attention.py が「30フレームのうちどの時間位置を参照しているか」を
見たのに対し、こちらは「各フレームの画像内で、どの空間位置(パッチ)に注目しているか」を
可視化・記録する。時間Attention解析で判明した「11番目のフレームへの集中は、映像内容
ではなくtime_embed由来の位置バイアスである」(verify_frame11_positional_bias.py)という
知見を踏まえ、11番目・30番目（自己/分類対象フレーム）・GTフェーズ切り替えフレームを
中心に、実際の画像に空間Attentionを重ねて表示する。

【このスクリプトが変更しないもの（安全性）】
analyze_temporal_attention.py と同様、既存の学習・評価・推論スクリプト・モデル定義・
Dataset・checkpoint・CSV・GT読み込みロジックは一切変更しない。すべてimportして
読み取り専用で使う。Attention重みの取得もforward hookのみで行う。

────────────────────────────────────────────────────────
【Temporal Attentionとの構造的な違い】
────────────────────────────────────────────────────────
models/timesformer.py の TimeSformerBlock.forward() 空間Attention部分:

    cls_expanded = cls_tok.expand(B, T, D).reshape(B * T, 1, D)
    s_in = patch_tok.reshape(B, T, S, D).reshape(B * T, S, D)
    s_in = torch.cat([cls_expanded, s_in], dim=1)      # (B*T, 1+S, D)
    s_out = self.spatial_attn(self.norm_spatial(s_in))

- クエリは自明: clsトークン（s_inの先頭, index=0）。分類ヘッドに渡るのはこのclsトークン
  由来の表現なので、「clsが各フレーム内のどこを見ているか」がそのまま分類根拠の
  空間的な手がかりになる（時間Attentionのときのような「どのフレームをqueryにするか」
  という選択の余地がない）。
- バッチ次元がB*T（本スクリプトはB=1固定なので、そのままT=30フレームに1:1対応する）。
  そのため時間Attentionのときのような「複数ウィンドウが混ざる」問題は起きず、
  1回のforwardで30フレーム全部の空間Attentionが素直に得られる。

Tensor shapeの流れ（1ウィンドウ=1回のforward, B=1固定）:
  spatial_attn への入力 x: (N=B*T=30, L=1+S=197, D=768)
  qkv(x) → reshape → (3, N=30, heads=12, L=197, head_dim=64) → q, k
  attn = softmax((q @ k^T) * scale) → (N=30, heads=12, 197 query, 197 key)
  cls→patch 行だけを取り出す: attn[:, :, 0, 1:] → (30フレーム, 12heads, 196patch)
  head方向に平均 → (30, 196) → 14x14グリッドにreshape

────────────────────────────────────────────────────────
【集約方針（曖昧さを排除するため明記）】
────────────────────────────────────────────────────────
1. query: clsトークン固定（上記の通り選択の余地がないため）
2. key: 196パッチ全部を14x14グリッドとして残す（集約しない。これがヒートマップそのもの）
3. head方向: 平均する（analyze_temporal_attention.pyと同じ方針）
4. layer方向: 平均しない。代表的な3層（デフォルト: 0, 5, 11＝初期・中間・最終）を
   並べて表示する。時間Attentionと異なり、層が深くなるほどパッチ表現は既に他フレーム・
   他パッチの情報と混ざっており、単純な層平均は「どの層の話か」を曖昧にしてしまうため。
   生データは全12層×30フレームぶん保存するので、他の層を見たい場合は raw/*.npy から
   後で再集計できる。
5. frame方向: 集約しない。可視化は「11番目・30番目(query/分類対象フレーム)・
   GTフェーズ切り替えフレーム（あれば）」を並べて表示する（時間Attention解析の結果、
   11番目は位置バイアスの可能性が高いと分かっているため、実際に画像のどこを見て
   いるのかを直接確認する狙い）。

────────────────────────────────────────────────────────
【Ground Truthのフェーズ境界判定】
────────────────────────────────────────────────────────
analyze_temporal_attention.py と同じロジック（gt_utils.true_phase_at + 1/3秒間隔での
フレーム時刻再構成）をそのまま再利用する。詳細はそちらのdocstring参照。

────────────────────────────────────────────────────────
【重要：Attentionの解釈について】
────────────────────────────────────────────────────────
analyze_temporal_attention.py と同じ注意点が適用される。空間Attentionが高い領域は
「モデルがその領域の特徴を強く混ぜ込んでいる」ことを示すが、「そこが分類の直接的な
根拠である」ことを保証するものではない。また、時間Attention解析で判明した通り、
Attentionのパターンには内容と無関係な学習済みバイアス（位置埋め込み等）が混入し得る
ため、空間Attentionについても同様のバイアスが存在する可能性を念頭に置くこと。

使い方:
    python analyze_spatial_attention.py [checkpoint_dir] [--layers 0,5,11]
        [--max-samples 5] [--only-errors] [--output-dir spatial_attention_analysis]
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

matplotlib.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from dataset_index.prepare_dataset import SurgicalPhaseDataset
from evaluation.test2 import load_eval_config, phases, phases_12
from utils.gt_utils import load_ground_truth_intervals, true_phase_at
from data_preparation.build_sliding_window_index import GT_CSV_TEMPLATE, VIDEO_DIR_TEMPLATE
from analyze_temporal_attention import (
    build_model,
    reconstruct_frame_times,
    detect_transition,
    ATTENTION_CAVEAT,
)

GRID_SIZE = 14  # img_size(224) / patch_size(16)


class SpatialAttentionRecorder:
    """
    TimeSformerの各ブロックの spatial_attn (_Attentionインスタンス) にforward hookを
    登録し、clsトークンから各パッチへのAttention重み（head平均後、14x14グリッド）を
    「読み取り専用」で記録するクラス。analyze_temporal_attention.py の
    TemporalAttentionRecorder と対になる構成（違いはdocstring「Temporal Attentionとの
    構造的な違い」参照）。モデル本体のforward()やパラメータは一切変更しない。
    """

    def __init__(self, model: torch.nn.Module):
        self.per_layer = {}  # layer_idx -> (T=30, 14, 14) tensor
        self._handles = []
        for layer_idx, block in enumerate(model.blocks):
            handle = block.spatial_attn.register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            # inputs[0] = norm_spatial適用後の入力 x, shape (N=B*T, L=1+S, D)
            # B=1固定前提なので N=T=30。_Attention.forward()と同じ計算をここで再現する。
            x = inputs[0]
            N, L, D = x.shape
            qkv = module.qkv(x).reshape(N, L, 3, module.num_heads, module.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, N, heads, L, head_dim)
            q, k = qkv[0], qkv[1]
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)  # (N=T, heads, 1+S, 1+S)

            cls_to_patch = attn[:, :, 0, 1:]  # clsクエリ行のみ。(T, heads, S=196)
            head_avg = cls_to_patch.mean(dim=1)  # (T, 196)
            grid = head_avg.reshape(-1, GRID_SIZE, GRID_SIZE)  # (T, 14, 14)
            self.per_layer[layer_idx] = grid.detach().to("cpu")

        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def stacked(self) -> torch.Tensor:
        """層0..depth-1の順に積んだ (depth, T, 14, 14) テンソルを返す。"""
        depth = len(self.per_layer)
        return torch.stack([self.per_layer[i] for i in range(depth)], dim=0)


def frame_tensor_to_image(frame_chw: torch.Tensor) -> np.ndarray:
    """SurgicalPhaseDatasetが返すフレーム(C,H,W, 0-1正規化済み、mean/std正規化はされていない
    ---dataset_index/prepare_dataset.py 参照)を、そのままmatplotlib表示用(H,W,C)に変換する。"""
    img = frame_chw.permute(1, 2, 0).numpy()
    return np.clip(img, 0.0, 1.0)


def overlay_heatmap(ax, frame_img: np.ndarray, heatmap_14x14: np.ndarray, title: str):
    """実フレーム画像に、14x14の空間Attentionヒートマップを224x224へ拡大して半透明重ね表示する。"""
    h, w = frame_img.shape[:2]
    heat = torch.from_numpy(heatmap_14x14).float().unsqueeze(0).unsqueeze(0)
    heat_up = F.interpolate(heat, size=(h, w), mode="bilinear", align_corners=False)
    heat_up = heat_up.squeeze().numpy()
    heat_norm = (heat_up - heat_up.min()) / (heat_up.max() - heat_up.min() + 1e-8)

    ax.imshow(frame_img)
    ax.imshow(heat_norm, cmap="inferno", alpha=0.45, extent=(0, w, h, 0))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def select_target_frames(sequence_length: int, has_transition: bool, transition_frame) -> list:
    """可視化の中心にするフレーム番号(1-indexed)を決める。
    方針: 11番目(位置バイアス疑いのフレーム)・30番目(query/分類対象フレーム)・
    GTフェーズ切り替えフレーム(あれば、かつ上記2つと重複しなければ追加)。"""
    targets = [11, sequence_length]
    if has_transition and transition_frame is not None and transition_frame not in targets:
        targets.append(int(transition_frame))
    return sorted(set(t for t in targets if 1 <= t <= sequence_length))


def main():
    parser = argparse.ArgumentParser(
        description="TimeSformerのSpatial Attentionを解析し、画像内の注目領域を可視化・記録する"
    )
    parser.add_argument(
        "checkpoint_dir", nargs="?",
        default="result/epochs_35/sliding_window_checkpoints_0914_10_46",
    )
    parser.add_argument("--checkpoint-name", default="best_model.pth")
    parser.add_argument("--csv", default="data/dataset_index_sliding_window_stride5.csv")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default="spatial_attention_analysis")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--only-errors", action="store_true", help="誤分類ウィンドウのみを対象にする")
    parser.add_argument(
        "--only-correct", action="store_true",
        help="正解ウィンドウのみを対象にする（--only-errorsとの同時指定不可。"
             "誤分類との比較用サンプルを集める場合に使う）",
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        help="走査順をシャッフルする。指定しない場合はCSVの並び順（症例ごとにまとまった順）"
             "のまま先頭からmax-samples件を探すため、症例が偏りやすい。--only-correctで"
             "無作為抽出したい場合はこれを付けることを推奨",
    )
    parser.add_argument("--seed", type=int, default=0, help="--shuffle時の乱数シード（再現性のため）")
    parser.add_argument(
        "--layers", default="0,5,11",
        help="可視化するTimeSformerBlockの層番号(0-indexed, カンマ区切り)。デフォルトは初期・中間・最終",
    )
    args = parser.parse_args()
    if args.only_errors and args.only_correct:
        raise ValueError("--only-errors と --only-correct は同時に指定できません")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_path = checkpoint_dir / args.checkpoint_name
    out_dir = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    raw_dir = out_dir / "raw"
    csv_dir = out_dir / "csv"
    for d in (plots_dir, raw_dir, csv_dir):
        d.mkdir(parents=True, exist_ok=True)

    layers_to_show = [int(x) for x in args.layers.split(",")]

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
    print(f"[analyze_spatial] {args.split} split: {len(dataset)} ウィンドウ (checkpoint: {checkpoint_path})")
    print(f"  表示対象レイヤー: {layers_to_show}")

    recorder = SpatialAttentionRecorder(model)
    gt_intervals_cache = {}

    def get_gt_intervals(case_id: str):
        if case_id not in gt_intervals_cache:
            gt_intervals_cache[case_id] = load_ground_truth_intervals(
                GT_CSV_TEMPLATE.format(case=case_id), active_phases
            )
        return gt_intervals_cache[case_id]

    rows = []
    n_analyzed = 0
    n_scanned = 0

    scan_order = list(range(len(dataset)))
    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(scan_order)

    with torch.no_grad():
        for idx in scan_order:
            if n_analyzed >= args.max_samples:
                break
            n_scanned += 1

            row_meta = dataset.data.iloc[idx]
            case_id = row_meta["case_id"]
            source_time_sec = float(row_meta["source_time_sec"])

            clip_tensor, label_tensor = dataset[idx]  # (30,3,224,224)
            true_label_id = int(label_tensor.item())
            true_label = active_phases[true_label_id]

            recorder.per_layer.clear()
            logits = model(clip_tensor.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
            pred_label_id = int(probs.argmax().item())
            pred_label = active_phases[pred_label_id]
            confidence = float(probs[pred_label_id].item())
            correct = pred_label_id == true_label_id

            if args.only_errors and correct:
                continue
            if args.only_correct and not correct:
                continue

            frame_times = reconstruct_frame_times(source_time_sec, sequence_length)
            gt_intervals = get_gt_intervals(case_id)
            frame_phases = [true_phase_at(gt_intervals, t) for t in frame_times]
            has_transition, transition_frame, n_transitions = detect_transition(frame_phases)

            spatial_stacked = recorder.stacked()  # (depth, T=30, 14, 14)
            target_frames = select_target_frames(sequence_length, has_transition, transition_frame)

            window_id = f"window_{n_analyzed + 1:04d}"
            plot_path = plots_dir / f"{window_id}_spatial.png"
            raw_path = raw_dir / f"{window_id}_spatial_attn.npy"
            np.save(raw_path, spatial_stacked.numpy())  # (depth=12, T=30, 14, 14) 全層・全フレーム保存

            # 先頭行に「元画像(ヒートマップなし)」を並べ、その下にレイヤーごとの重ね表示を
            # 並べる。同じ列(同じフレーム)で縦に比較できるので、ヒートマップが元の映像の
            # どこに対応するか一目で分かるようにするため。
            n_rows, n_cols = len(layers_to_show) + 1, len(target_frames)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.1 * n_rows), squeeze=False)
            metric_rows = {}

            def frame_role_label(frame_1idx: int) -> str:
                role = ""
                if frame_1idx == sequence_length:
                    role = "(query)"
                elif frame_1idx == 11:
                    role = "(位置バイアス疑い)"
                if has_transition and frame_1idx == transition_frame:
                    role += "(切替後)"
                return role

            for c, frame_1idx in enumerate(target_frames):
                frame_0idx = frame_1idx - 1
                frame_img = frame_tensor_to_image(clip_tensor[frame_0idx])
                ax_raw = axes[0][c]
                ax_raw.imshow(frame_img)
                ax_raw.set_title(f"frame{frame_1idx}{frame_role_label(frame_1idx)}\n(元画像)", fontsize=9)
                ax_raw.set_xticks([])
                ax_raw.set_yticks([])

            for r, layer_idx in enumerate(layers_to_show, start=1):
                for c, frame_1idx in enumerate(target_frames):
                    frame_0idx = frame_1idx - 1
                    heatmap = spatial_stacked[layer_idx, frame_0idx].numpy()  # (14,14)
                    frame_img = frame_tensor_to_image(clip_tensor[frame_0idx])
                    title = f"layer{layer_idx} / frame{frame_1idx}{frame_role_label(frame_1idx)}"
                    overlay_heatmap(axes[r][c], frame_img, heatmap, title)

                    flat = heatmap.flatten()
                    argmax_1d = int(np.argmax(flat))
                    metric_rows[f"L{layer_idx}_f{frame_1idx}_argmax_row"] = argmax_1d // GRID_SIZE
                    metric_rows[f"L{layer_idx}_f{frame_1idx}_argmax_col"] = argmax_1d % GRID_SIZE
                    metric_rows[f"L{layer_idx}_f{frame_1idx}_max_value"] = round(float(flat.max()), 5)
                    metric_rows[f"L{layer_idx}_f{frame_1idx}_concentration"] = round(
                        float(flat.max() / (flat.mean() + 1e-8)), 3
                    )  # 1.0に近いほど一様、大きいほど特定パッチに集中

            result_word = "正解" if correct else "誤分類"
            result_color = "#0b6e67" if correct else "#b23b2f"
            fig.suptitle(
                f"{window_id}  {case_id}  t={source_time_sec:.1f}s  "
                f"正解:{true_label} 予測:{pred_label}({confidence*100:.1f}%) [{result_word}]",
                fontsize=11, color=result_color,
            )
            fig.text(0.01, 0.005, ATTENTION_CAVEAT, fontsize=7.5, color="#666666")
            fig.tight_layout(rect=(0, 0.02, 1, 0.95))
            fig.savefig(plot_path, dpi=130)
            plt.close(fig)

            row = {
                "window_id": window_id,
                "case_id": case_id,
                "source_case_video": VIDEO_DIR_TEMPLATE.format(case=case_id),
                "window_end_sec": round(source_time_sec, 3),
                "true_label": true_label,
                "predicted_label": pred_label,
                "correct": correct,
                "confidence": round(confidence, 4),
                "has_transition": has_transition,
                "transition_frame": transition_frame if transition_frame is not None else "",
                "target_frames_shown": ",".join(str(f) for f in target_frames),
                "layers_shown": args.layers,
                "plot_path": str(plot_path).replace("\\", "/"),
                "raw_attn_path": str(raw_path).replace("\\", "/"),
                **metric_rows,
            }
            rows.append(row)
            n_analyzed += 1
            status = "誤分類" if not correct else "正解"
            print(f"  [{n_analyzed}/{args.max_samples}] {window_id} {case_id} t={source_time_sec:.1f}s "
                  f"true={true_label} pred={pred_label} ({status}) target_frames={target_frames}")

    recorder.remove()

    summary_csv = csv_dir / "spatial_attention_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    readme_path = out_dir / "README.md"
    readme_path.write_text(_build_readme(args, layers_to_show, len(dataset), n_scanned, n_analyzed, checkpoint_path), encoding="utf-8")

    print(f"\n完了。{n_analyzed}件のウィンドウを解析しました（{n_scanned}件走査）。")
    print(f"  サマリーCSV: {summary_csv}")
    print(f"  プロット: {plots_dir}/")
    print(f"  生Attentionデータ: {raw_dir}/")
    print(f"  説明: {readme_path}")


def _build_readme(args, layers_to_show, dataset_size, n_scanned, n_analyzed, checkpoint_path) -> str:
    return f"""# Spatial Attention 解析結果

`analyze_spatial_attention.py` の出力です。既存スクリプト・モデル・Datasetは一切
変更していません（forward hookによる読み取り専用のAttention取得。詳細はスクリプト
冒頭のdocstring参照）。analyze_temporal_attention.py / verify_frame11_positional_bias.py
と対になる解析です。

## 実行条件
- checkpoint: `{checkpoint_path}`
- インデックスCSV: `{args.csv}` (split={args.split})
- 表示レイヤー: {layers_to_show}（0-indexed。0=初期層, 11=最終層）
- 対象split内の総ウィンドウ数: {dataset_size}
- 走査/解析したウィンドウ数: {n_scanned} / {n_analyzed}
- only_errors: {args.only_errors}

## ディレクトリ構成
```
{args.output_dir}/
├── plots/window_XXXX_spatial.png  layer×frameのグリッド。実フレームにヒートマップ重ね表示
├── raw/window_XXXX_spatial_attn.npy  (depth=12, T=30, 14, 14) 全層・全フレームの生データ
└── csv/spatial_attention_summary.csv  1行=1ウィンドウ。表示した(layer,frame)組ごとの
    argmax位置(14x14グリッド内のrow,col)・最大値・集中度(max/mean比)を記録
```

## 可視化対象フレームの選び方
11番目(時間Attention解析で位置バイアスの疑いが強いフレーム)・30番目(query/分類対象
フレーム)・GTフェーズ切り替え直後のフレーム(あれば)を並べて表示します。

## Attentionの解釈について
{ATTENTION_CAVEAT}
時間Attention解析(verify_frame11_positional_bias.py)で、Attentionパターンには内容と
無関係な学習済みバイアスが混入し得ることが確認されています。空間Attentionについても、
同様のバイアス(例えば画像中央や特定パッチ位置への固定的な偏り)が無いか、複数ウィンドウを
比較して確認することを推奨します。
"""


if __name__ == "__main__":
    main()
