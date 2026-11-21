# OnlineTAS（Zhong, Ding, Yao, NeurIPS 2024）

- 引用キー：2024_zhong_onlinetas
- 著者：工学系（University of Adelaide、National University of Singapore。PDF で確認）
- 発表：NeurIPS 2024 ／ arXiv 2024-11（2411.01122）
- リンク：https://arxiv.org/html/2411.01122
- ライセンス：arXiv 標準（nonexclusive-distrib 1.0）→ 全文和訳は不可、作るなら日本語の要約
- 状態：未読（日本語要約あり、2026-11-21。PDF は本文 10 ページ＋付録）。11-17 の読書メモ（planB-baseline-reproduce worktree の notes/2026-11-17_flicker_delay_papers.md）から移したもの。arXiv の HTML で本文を確認した範囲。

## 一言で
自信と継続時間による後処理で edit が大きく上がるが、正解率は少し下がる。手術ではなく料理動画など。

## 何をしたか
- Breakfast、50Salads、GTEA のオンライン行動区切り。
- 後処理（Algorithm 2）：自信 q_t = max softmax < θ で、前のラベルから変わってまだ l_min（= σ × train の最長動画の長さ）未満なら、前のラベルを保つ。

## 結果
- edit が大きく上がり、正解率は 2〜3 ポイント下がる（例：50Salads で edit 28.8 → 71.5）。
- θ と σ の ablation は Table 6・7。

## 自分の研究との関係
- 関係するテーマ：揺れと遅れ
- 遅れは測っていない。仕組みとしては、自信の低い本当の切り替わりは待たされるはず。

## 要約を作るときに気づいたこと（2026-11-21）
- 後処理の ℓ_min = σ × 学習データの最長の動画の長さ（σ = 1/16 が既定）。θ を上げるほど区間の指標が上がり Acc が下がる（表 6）。
- 著者自身が失敗例として「新しい行動の始まりの検出が遅れる」「確信度が高い誤りは直らない」と書いている（p.9）。遅れの数値はない。
- 表 1 と表 8・9 で同じ名前の行の値が食い違う（原文の不整合）。
