# OnlineTAS（Zhong, Ding, Yao, NeurIPS 2024）

- 引用キー：2024_zhong_onlinetas
- 著者：工学系（コンピュータビジョン）
- 発表：NeurIPS 2024 ／ arXiv 2024-11（2411.01122）
- リンク：https://arxiv.org/html/2411.01122
- 状態：未取得（PDFなし）。11-17 の読書メモ（planB-baseline-reproduce worktree の notes/2026-11-17_flicker_delay_papers.md）から移したもの。arXiv の HTML で本文を確認した範囲。

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
