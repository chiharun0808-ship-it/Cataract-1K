# Data-Efficient Surgical Phase Segmentation in Small-Incision Cataract Surgery: A Controlled Study of Vision Foundation Models

- 引用キー：2026_spencer_dataefficient
- 著者：Lincoln Spencer、Song Wang、Chen Chen（University of Central Florida, Institute of Artificial Intelligence）。工学系（医者の共著者はなし）
- 発表：arXiv 2026-04（2604.10514、v1）。会議・雑誌は書かれていない
- リンク：https://arxiv.org/abs/2604.10514 ／ https://arxiv.org/html/2604.10514 ／ https://sl2005.github.io/DataEfficient-sics-phase-seg/
- ライセンス：CC BY 4.0（arXiv）→ 全文和訳できる
- 状態：未読（和訳あり、2026-11-21。7 ページ）。2026-11-20 に arXiv の HTML（v1）を確認した範囲のメモ（取得は要約を経由したので、数字は PDF で確かめる）。

## 一言で
小切開白内障手術（SICS-155）で、凍結した画像・動画の基盤モデルの特徴を同じ MS-TCN++ に入れて比べた。DINOv3 ViT-7B が最良、ViT-L もほぼ同じ。

## 何をしたか（データ・手法・評価）
- SICS-155 の train/val 115 本（19 フェーズ）、5 分割の交差検証。
- 比べた特徴：ResNet-50、I3D、DINOv3 ViT-B/16・L/16・7B/16（224 入力）、V-JEPA2 ViT-L・ViT-g。
- 1 フレーム 1 特徴（"pooled representation"。CLS かパッチの平均かは書いていない）。上に MS-TCN++（全部同じ設定）。
- CataractFT：**Cataract-1K の動画 1,000 本**（ラベルなし）で自己教師ありの追加学習＋LoRA（Cataract-1K 56 本＋Cataract-101）。

## 結果（数字、macro F1 %）
- DINOv3 B 71.2、L 75.8、7B 76.5。I3D 71.7、V-JEPA2 L 69.9、g 67.5。
- CataractFT：ViT-B は +2.9、ViT-L は −1.8、V-JEPA2 L は −8.1。「効果はまちまちで、強く悪くなることもある」。
- 7B の重みだけで fp16 で約 13.4 GB。B200 1 枚で実験。

## 自分の研究との関係
- なぜ読む・何に使うか：2026-11-20 の「長い窓の枝」の DINOv2 を DINOv3 に替える根拠。白内障で DINOv3 が効くこと、L と 7B の差が小さいこと。
- 注意：CataractFT の重みは Cataract-1K（test の症例を含む）で学習しているので、公開されても自分の実験には使えない。

## 疑問点・あとで確認すること
- 特徴が CLS かパッチの平均か。コードと重みの公開の有無。

## 和訳を作るときに気づいたこと（2026-11-21）
- 特徴は "pooled representation" とだけあり、CLS かパッチの平均かは PDF にも書かれていない。
- MS-TCN++ の損失に平滑化の MSE（重み 0.35）が入っている。
- CataractFT＋LoRA は自己教師ありの継続だけに比べて edit が下がる（ViT-B −1.29、ViT-L −1.80）＝フレームの識別が鋭くなると境界で揺れが増える、という観察（p.6）。
