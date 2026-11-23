# Cataract-LMM: Large-Scale, Multi-Source, Multi-Task Benchmark for Deep Learning in Surgical Video Analysis

- 引用キー：2026_ahmadi_cataractlmm
- 著者：Ahmadi MJ, Gandomi I, Abdi P, Mohammadi S-F, Taslimi A, Khodaparast M, Hashemi H, Tavakoli M, Taghirad HD（K. N. Toosi 工科大学 と Farabi 眼科病院・Noor 眼科病院〔テヘラン医科大学〕、Alberta 大学）。**混合**
- 発表：2026-05（Scientific Data 13:1189、2026-05-23 公開）／ arXiv 2025-10（2510.16371）
- リンク：https://doi.org/10.1038/s41597-026-07464-0 ／ PMC13473647 ／ https://arxiv.org/abs/2510.16371
- ライセンス：CC BY-NC-ND 4.0 → 全文和訳はできない（作るなら日本語の要約）
- データ：Hugging Face `mjahmadi/Cataract-LMM`。**2026-11-23 も "disabled": true でダウンロード不可**（2026-11-06 に確認したときと同じ状況）
- 状態：未読（2026-11-24 登録）

## 一言で
2 施設（Farabi 2,930 本・Noor 70 本）の白内障手術 3,000 本。工程・領域分割・器具と組織の追跡・技能の点数の 4 種のラベル。**片方の施設で学習してもう片方で試す**ベースラインがある。

## 結果（本文 p.11–12）
- 工程ラベルは 150 本（Farabi 129・Noor 21）。学習・検証は Farabi だけ、テストは Farabi 23 本（同じ施設）と Noor 21 本（別の施設）。
- 顕微鏡が施設で違う：Farabi は 720×480・30 fps、Noor は ZEISS ARTEVO 800 で 1920×1080・60 fps（p.3）。
- 別の施設で性能が落ちる：微調整した MViT-B（オンライン）は F1 77.1% → 57.6%。凍結した DINO を使う DINO-ASFormer（オフライン）は 79.98% → 67.9%（落ち幅が小さい）。CLIP を使うモデルも安定。
- 最も難しい工程は Capsule Polishing（隣の工程と見た目が似ている）。

## 自分の研究との関係
- 研究の方針（施設間の汎化）の中心のデータの元論文。施設が変わると落ちること、凍結した基盤モデルが強いことを数字で示している。
- オンライン（因果的）とオフラインの両方の結果がある。
