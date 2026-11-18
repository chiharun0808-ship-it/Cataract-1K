# Stabilizing Temporal Inference Dynamics for Online Surgical Phase Recognition（Liu ら, MICCAI 2026）

- 引用キー：liu2026stabilizing
- 著者：未確認
- 発表：MICCAI 2026（開催月は未確認）／ arXiv 2026-05（2605.16387）
- リンク：https://arxiv.org/html/2605.16387
- 状態：未取得（PDFなし）。11-17 の読書メモ（planB-baseline-reproduce worktree の notes/2026-11-17_flicker_delay_papers.md）から移したもの。arXiv の HTML で本文を確認した範囲。

## 一言で
TFI の出典。証拠を累積して閾値を超えたときだけ切り替える。遅れは測っていない。

## 何をしたか
- EGTP：候補フェーズの証拠（log 確率の差）を片側に累積し、正規化した値が閾値 k を超えたときだけ切り替える。
- TEC loss：誤りの始まりの部分の損失を重くする。
- Cholec80、AutoLaparo。Trans-SVNet、SKiT、Surgformer に適用。

## 結果
- k を大きくすると TFI は下がり、正解率は少し上がってから下がる（Fig. 3。数字は図のみ）。

## 自分の研究との関係
- 問い：Q3
- 遅れは測っていない。「k が大きすぎると、本当の切り替わりを遅らせたり見逃したりしうる」と一文あるだけ。

## 疑問点・あとで確認すること
- k を選んだ分割が書かれていない。本文と Table 3 で k の値が食い違う。
- 著者名と所属（refs.bib の TODO）。
