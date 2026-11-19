# 論文一覧

ルールとタグの意味は [README.md](README.md)。行はキーの順（＝発表年の順）に並べる。

| キー | 題名（短く） | 発表 | 会議・雑誌 | 著者の立場 | テーマ | 章 | 問い | 状態 | 一言 |
|---|---|---|---|---|---|---|---|---|---|
| [2024_ghamsarian_cataract1k](2024_ghamsarian_cataract1k.md) | Cataract-1K Dataset for Deep-Learning-Assisted Analysis of Cataract Surgery Videos | 2024-04 | Scientific Data | 混合 | データセット | 関連研究 | Q1, Q4 | 未読 | 使っているデータセットの元論文。13 工程・56 本のラベルの定義を確認する |
| [2024_zhong_onlinetas](2024_zhong_onlinetas.md) | OnlineTAS | 2024-12（NeurIPS）／arXiv 2024-11 | NeurIPS 2024 | 工学系 | 揺れ, 手法一般 | 関連研究 | Q3 | 未取得 | 自信と継続時間による後処理で edit が大きく上がる。手術以外の動画 |
| [2025_shah_stepal](2025_shah_stepal.md) | StepAL: Step-aware Active Learning for Cataract Surgical Videos | 2025-??（MICCAI）／arXiv 2025-07 | MICCAI 2025 | 混合（推測） | 手法一般, データセット | 関連研究 | Q1 | 未読 | Cataract-1K で手順認識の能動学習。動画単位でラベル付け対象を選ぶ |
| [2025_tbd_depthor](2025_tbd_depthor.md) | 深度カメラの手術室動画のフェーズ認識 | 2025-?? | Surgical Endoscopy | 未確認 | 揺れ, 遅れ | 関連研究 | Q3 | 未取得 | 医学誌。予測を遅らせて平滑化すると頑健になるらしい（本文未読） |
| [2026_ali_labels](2026_ali_labels.md) | Labels for automated step recognition in cataract surgery for videos in the Cataract-1K dataset | 2026-10 | Scientific Data | 混合 | データセット, 揺れ, 評価指標 | 関連研究, 評価 | Q1, Q3, Q4 | 流し読み（Claude、和訳あり） | Cataract-1K 全 1,000 本に 10 手順のラベル。386 本は手作業で境界の基準を明示（ICC 0.994〜1.000）。疑似ラベルの揺れは 5 フレームで平滑化 |
| [2026_guo_ttdf](2026_guo_ttdf.md) | TTDF | 2026-09（arXiv） | arXiv | 工学系 | 揺れ, 遅れ | 手法 | Q3 | 未取得 | 遅れを数字で出している唯一の論文。余計な切り替わりを減らすと遅れの中央値が 0 → 6.3 |
| [2026_liu_stabilizing](2026_liu_stabilizing.md) | Stabilizing Temporal Inference Dynamics for Online Surgical Phase Recognition | 2026-??（MICCAI）／arXiv 2026-05 | MICCAI 2026 | 未確認 | 揺れ, 評価指標 | 手法, 評価 | Q3 | 未取得 | TFI の出典。証拠を累積して閾値を超えたら切り替える。遅れは測っていない |
| [2026_mezzina_framework](2026_mezzina_framework.md) | A Framework for Real-Time Surgical Phase Recognition … | 2026-??（WACV） | WACV 2026 | 未確認 | 遅れ, 揺れ | 手法 | Q3 | 未取得 | 前向きだけの Viterbi で正確さと遅れを釣り合わせるらしい（本文未読） |
| [2026_tjokrovonco_artificial](2026_tjokrovonco_artificial.md) | AI across the cataract care continuum（CAICC スコーピングレビュー） | 2026-10 | BMC Ophthalmology | 医者 | 汎化, 臨床用途, 評価指標 | 背景 | Q2, Q4 | 流し読み | 白内障 AI 185 本のうち 68% が内部検証どまり、多施設の前向き評価は 1%。IEEE は未検索 |
| [tbd_surgcbvr](tbd_surgcbvr.md) | SurgCBVR-CAT（クリップ検索ベンチマーク） | 未出版 | （Data in Brief 予定） | 未確認 | データセット | 関連研究 | — | 未読 | Cataract-1K の工程ラベルから 6,487 クリップ。論文が出たら登録し直す |
