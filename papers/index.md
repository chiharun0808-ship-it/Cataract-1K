# 論文一覧

ルールとタグの意味は [README.md](README.md)。行はキーの順（＝発表年の順）。

## 読んだ論文

| キー | 題名（短く） | 発表 | 会議・雑誌 | 著者の立場 | テーマ | なぜ読む・何に使う | 状態 |
|---|---|---|---|---|---|---|---|
| [2026_ali_labels](2026_ali_labels.md) | Labels for automated step recognition in cataract surgery for videos in the Cataract-1K dataset | 2026-10 | Scientific Data | 混合 | ラベル・データセット, 揺れ, 評価方法 | 「揺れは正解ラベルが大まかなせい？」を確かめる根拠。境界の基準を決めると人どうしのラベルはほぼ一致（ICC 0.994〜1.000）。疑似ラベルの揺れは 5 フレームで平滑化 | 読んだ（和訳あり） |
| [2026_tjokrovonco_artificial](2026_tjokrovonco_artificial.md) | AI across the cataract care continuum（CAICC スコーピングレビュー） | 2026-10 | BMC Ophthalmology | 医者 | 臨床（医者の視点）, 汎化 | 研究の背景の根拠。白内障 AI の 68% が内部検証どまりで多施設の前向き評価は 1%。医者側が「性能が高くても他施設で使える・役立つとは限らない」と言っている | 読んだ（和訳あり） |

## 読む候補

まだ読んでいない論文。読んだら上の表に移す。

| キー | 題名（短く） | 発表 | 会議・雑誌 | 著者の立場 | テーマ | なぜ読む・何に使う | 状態 |
|---|---|---|---|---|---|---|---|
| [2024_ghamsarian_cataract1k](2024_ghamsarian_cataract1k.md) | Cataract-1K Dataset for Deep-Learning-Assisted Analysis of Cataract Surgery Videos | 2024-04 | Scientific Data | 混合 | ラベル・データセット | 使っているデータの元論文。13 工程・56 本のラベルの定義と付け方を確認する | 未読（PDF なし） |
| [2024_zhong_onlinetas](2024_zhong_onlinetas.md) | OnlineTAS | 2024-12（NeurIPS）／arXiv 2024-11 | NeurIPS 2024 | 工学系 | 揺れ, 手法一般 | 自信度と継続時間による後処理で揺れを抑える方法。遅れは測っていない（比較相手） | 未取得 |
| [2025_shah_stepal](2025_shah_stepal.md) | StepAL: Step-aware Active Learning for Cataract Surgical Videos | 2025-??（MICCAI）／arXiv 2025-07 | MICCAI 2025 | 混合（推測） | ラベル・データセット, 手法一般 | Cataract-1K を手順として扱った先行研究。ラベル付けの負担を減らす方法（2026_ali_labels と同じグループ） | 未読（PDF なし） |
| [2025_tbd_depthor](2025_tbd_depthor.md) | 深度カメラの手術室動画のフェーズ認識 | 2025-?? | Surgical Endoscopy | 未確認 | 揺れ, 遅れ | 医学誌の立場から「予測を遅らせて平滑化すると頑健」と言っているらしい。揺れと遅れの関係の手がかり | 未取得（ログインが必要） |
| [2026_guo_ttdf](2026_guo_ttdf.md) | TTDF | 2026-09（arXiv） | arXiv | 工学系 | 揺れ, 遅れ | 遅れを数字で出した唯一の論文（余計な切り替わりを減らすと遅れの中央値 0 → 6.3）。揺れと遅れのトレードオフの比較相手 | 未取得 |
| [2026_liu_stabilizing](2026_liu_stabilizing.md) | Stabilizing Temporal Inference Dynamics for Online Surgical Phase Recognition | 2026-??（MICCAI）／arXiv 2026-05 | MICCAI 2026 | 未確認 | 揺れ, 評価方法 | 揺れの指標 TFI の出典。証拠を累積して切り替える方法。遅れは測っていない | 未取得 |
| [2026_mezzina_framework](2026_mezzina_framework.md) | A Framework for Real-Time Surgical Phase Recognition … | 2026-??（WACV） | WACV 2026 | 未確認 | 揺れ, 遅れ | 前向きだけの Viterbi で正確さと遅れを釣り合わせるらしい。遅れを扱う数少ない論文 | 未取得（403） |
| [tbd_surgcbvr](tbd_surgcbvr.md) | SurgCBVR-CAT（クリップ検索ベンチマーク） | 未出版 | （Data in Brief 予定） | 未確認 | ラベル・データセット | Cataract-1K の工程ラベルを使った派生資源。論文が出たら登録し直す | 論文がまだない |
