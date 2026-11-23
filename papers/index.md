# 論文一覧

ルールとタグの意味は [README.md](README.md)。行はキーの順（＝発表年の順）。

まとめ：[眼科医から見た白内障 AI の現状と今後の研究課題、試せるデータセット（2026-11-23）](_synthesis_clinical_ai.md)

## 読んだ論文

| キー | 題名（短く） | 発表 | 会議・雑誌 | 著者の立場 | テーマ | なぜ読む・何に使う | 状態 |
|---|---|---|---|---|---|---|---|
| [2026_ali_labels](2026_ali_labels.md) | Labels for automated step recognition in cataract surgery for videos in the Cataract-1K dataset | 2026-10 | Scientific Data | 混合 | ラベル・データセット, 揺れ, 評価方法 | 「揺れは正解ラベルが大まかなせい？」を確かめる根拠。境界の基準を決めると人どうしのラベルはほぼ一致（ICC 0.994〜1.000）。疑似ラベルの揺れは 5 フレームで平滑化 | 読んだ（和訳あり） |
| [2026_tjokrovonco_artificial](2026_tjokrovonco_artificial.md) | AI across the cataract care continuum（CAICC スコーピングレビュー） | 2026-10 | BMC Ophthalmology | 医者 | 臨床（医者の視点）, 汎化 | 研究の背景の根拠。白内障 AI の 68% が内部検証どまりで多施設の前向き評価は 1%。医者側が「性能が高くても他施設で使える・役立つとは限らない」と言っている | 読んだ（和訳あり） |

## 読む候補

まだ読んでいない論文。読んだら上の表に移す。

| キー | 題名（短く） | 発表 | 会議・雑誌 | 著者の立場 | テーマ | なぜ読む・何に使う | 状態 |
|---|---|---|---|---|---|---|---|
| [2019_morita_realtime](2019_morita_realtime.md) | Real-Time Extraction of Important Surgical Phases in Cataract Surgery Videos | 2019-11 | Scientific Reports | 混合 | 揺れ, 遅れ, 評価方法 | ツカザキ病院の工程のリアルタイム抽出。移動平均の窓を長くすると遅れるので 5 フレームにした、と揺れと遅れのトレードオフを明記。境界の誤差を秒で評価（平均 5.25 秒） | 未読 |
| [2020_morita_realtime](2020_morita_realtime.md) | Real-Time Surgical Problem Detection and Instrument Tracking in Cataract Surgery | 2020-11 | J Clin Med | 混合 | 臨床（医者の視点）, 遅れ | 日本の企業・工学系と眼科医（田淵）によるリアルタイムの問題検出。医者がリアルタイムで何を知りたいか（警告・危険度）の材料 | 未読（和訳あり） |
| [2021_bertasius_space](2021_bertasius_space.md) | Is Space-Time Attention All You Need for Video Understanding?（TimeSformer） | 2021-07（ICML）／arXiv 2021-02 | ICML 2021 | 工学系 | 手法一般 | 自分の工程認識モデルの元論文。分割時空間注意（時間注意は同じ位置のパッチどうし）、8 フレーム×1/32 間隔の設計を引用するときの出典 | 未読（日本語要約あり） |
| [2021_oshika_aieye](2021_oshika_aieye.md) | 『AIの眼』による医療安全確保に関する研究 令和2年度報告書（厚労科研） | 2021-05 | 厚生労働科学研究 報告書 | 混合 | 臨床（医者の視点）, ラベル・データセット | 国の研究班（大鹿班）の 1 年目。学会主導の動画収集と、工程分割・手術時間推定の始まり | 未読 |
| [2021_yeh_phacotrainer](2021_yeh_phacotrainer.md) | PhacoTrainer: A Multicenter Study of Deep Learning for Activity Recognition in Cataract Surgical Videos | 2021 | Transl Vis Sci Technol | 混合（推測） | 汎化, ラベル・データセット | 複数施設での工程認識の先行研究。施設の違いで性能がどう変わるかを確認する | 未読（PDF なし） |
| [2022_fang_global](2022_fang_global.md) | Global–local multi-stage TCN for cataract surgery phase recognition（GL-MSTCN） | 2022-11 | BioMedical Engineering OnLine | 混合 | 手法一般 | 白内障の工程認識で、瞳孔・器具・フレーム全体の 3 つの流れを使う。器具の先端の周りを見る案の先行例。Cataract-101 で 96.5% | 未読 |
| [2022_kiuchi_deep](2022_kiuchi_deep.md) | Deep Learning-Based System for Preoperative Safety Management in Cataract Surgery | 2022-09 | J Clin Med | 医者 | 汎化, 臨床（医者の視点） | ツカザキ病院で作った取り違え防止 AI を筑波大病院に導入。1 回目の認証成功率が開発元より下がった（眼内レンズ 89.0%→67.4%）＝施設が変わると性能が落ちる実例 | 未読 |
| [2022_oshika_aieye](2022_oshika_aieye.md) | 『AIの眼』による医療安全確保に関する研究 令和3年度報告書（厚労科研） | 2022-05 | 厚生労働科学研究 報告書 | 混合 | 臨床（医者の視点）, ラベル・データセット, 汎化 | 医師が動画に工程名を付けている（将来は工程認識 AI に置き換える予定）。術者ごとに約 20 例で調整すると手術時間の推定が良くなる | 未読 |
| [2022_tabuchi_realtime](2022_tabuchi_realtime.md) | Real-time AI evaluation of cataract surgery: a preliminary study | 2022-04 | Taiwan J Ophthalmol | 混合 | 臨床（医者の視点） | 手術中にリアルタイムで出す危険度（0〜1）で研修医と指導医を比較。医師面談の「リアルタイム警告・危険度」の用途の実例 | 未取得（PDF なし） |
| [2023_ding_less](2023_ding_less.md) | Less Is More: Surgical Phase Recognition From Timestamp Supervision | 2023-06／arXiv 2022-02 | IEEE TMI | 混合 | ラベル・データセット, 揺れ | 工程の中の 1 点だけのラベルで学習（付ける時間 74% 減）。境界付近のあいまいなラベルを除くと良くなる＝「揺れはラベルが大まかなせい？」の材料 | 未読 |
| [2023_oshika_aieye](2023_oshika_aieye.md) | 『AIの眼』による医療安全確保に関する研究 令和4年度報告書（厚労科研） | 2023-?? | 厚生労働科学研究 報告書 | 混合 | 臨床（医者の視点）, 汎化 | AI を他施設に広げるには作り直しが必要（施設で流れが違う）＝汎化の医師側の根拠。0.5 秒から出すリアルタイムの危険度（研修医 対 スタッフ医師） | 未読 |
| [2023_shah_glsformer](2023_shah_glsformer.md) | GLSFormer: Gated - Long, Short Sequence Transformer for Step Recognition in Surgical Videos | 2023-10（MICCAI）／arXiv 2023-07 | MICCAI 2023 | 混合 | 揺れ, 手法一般 | TimeSformer に長い系列をゲートで足すと白内障の手順認識が上がる（Jaccard 75.97→81.89）。長い窓の比較相手。そのまま足すと悪化する点に注意 | 未読（日本語要約あり） |
| [2023_wang_realtime](2023_wang_realtime.md) | Real-time estimation of the remaining surgery duration for cataract surgery | 2023-05 | BMC Med Inform Decis Mak | 混合 | 汎化, 臨床（医者の視点） | 手術の残り時間のリアルタイム推定。主な術者で学習→少しの微調整で他の術者へ（MAE 19.4 秒→他の術者 28〜31 秒）。術者の違いによる汎化の例 | 未読 |
| [2024_ghamsarian_cataract1k](2024_ghamsarian_cataract1k.md) | Cataract-1K Dataset for Deep-Learning-Assisted Analysis of Cataract Surgery Videos | 2024-04 | Scientific Data | 混合 | ラベル・データセット | 使っているデータの元論文。13 工程・56 本のラベルの定義と付け方を確認する | 未読（和訳あり） |
| [2024_li_deep](2024_li_deep.md) | Deep learning for surgical workflow analysis: a survey of progresses, limitations, and trends | 2024-09 | Artificial Intelligence Review | 混合 | 手法一般, 評価方法 | 工程認識の総説。後処理・評価指標・データセットの整理を関連研究の章に使う | 未読 |
| [2024_oquab_dinov2](2024_oquab_dinov2.md) | DINOv2: Learning Robust Visual Features without Supervision | 2024-01（TMLR）／arXiv 2023-04 | TMLR | 工学系 | 手法一般 | 長い窓の枝の特徴抽出器の出典。凍結のまま線形層だけで CLIP 系と互角、微調整の伸びは小さい | 未読（日本語要約あり） |
| [2024_satyanaik_optimizing](2024_satyanaik_optimizing.md) | Optimizing latent graph representations of surgical scenes for unseen domain generalization（LG-DG） | 2024-04 | Int J CARS | 混合 | 汎化 | 手術画面を器具・解剖のグラフで表すと、学習に使っていない施設でも強い（腹腔鏡、仏→中国）。汎化の手法の例 | 未読 |
| [2024_tabuchi_largescale](2024_tabuchi_largescale.md) | Large-scale observational study of AI-based patient and surgical material verification system（37,529 件） | 2024-11 | BMJ Qual Saf | 医者 | 臨床（医者の視点）, 汎化 | 取り違え防止 AI の実臨床。誤りは AI を使わなかった場面と、未学習の新しいレンズ（AI の更新の遅れ）で起きた。性能より運用が大事という医師側の主張 | 未読 |
| [2024_zhong_onlinetas](2024_zhong_onlinetas.md) | OnlineTAS | 2024-12（NeurIPS）／arXiv 2024-11 | NeurIPS 2024 | 工学系 | 揺れ, 手法一般 | 自信度と継続時間による後処理で揺れを抑える方法。遅れは測っていない（比較相手） | 未読（日本語要約あり） |
| [2025_li_multiview](2025_li_multiview.md) | Multi-View Test-Time Adaptation for Semantic Segmentation in Clinical Cataract Surgery（MUTA） | 2025-05 | IEEE TMI | 工学系 | 汎化 | 白内障の領域分割で、施設が変わったときにその場でモデルを合わせる（テスト時適応）。白内障の施設間のずれを直接扱った例 | 未取得（PDF なし） |
| [2025_sachdeva_utility](2025_sachdeva_utility.md) | Utility of an LLM-powered experts-in-the-loop chatbot for cataract surgery patients（CataractBot） | 2025-12 | Eur J Ophthalmol | 混合 | 臨床（医者の視点） | 白内障の術前後の患者の質問に答える LLM チャットボットを、医師が後から確認する運用。工程認識とは別の、白内障診療の AI の背景 | 未取得（PDF なし） |
| [2025_shah_stepal](2025_shah_stepal.md) | StepAL: Step-aware Active Learning for Cataract Surgical Videos | 2025-??（MICCAI）／arXiv 2025-07 | MICCAI 2025 | 混合 | ラベル・データセット, 手法一般 | Cataract-1K を手順として扱った先行研究。ラベル付けの負担を減らす方法（2026_ali_labels と同じグループ） | 未読（和訳あり） |
| [2025_simeoni_dinov3](2025_simeoni_dinov3.md) | DINOv3 | 2025-08（arXiv） | arXiv | 工学系 | 手法一般 | DINOv2 から替える候補。Gram anchoring で密な特徴が安定、動画の追跡も良い（時間方向に安定した特徴） | 未読（日本語要約あり） |
| [2025_tbd_depthor](2025_tbd_depthor.md) | 深度カメラの手術室動画のフェーズ認識 | 2025-?? | Surgical Endoscopy | 未確認 | 揺れ, 遅れ | 医学誌の立場から「予測を遅らせて平滑化すると頑健」と言っているらしい。揺れと遅れの関係の手がかり | 未取得（ログインが必要） |
| [2026_ahmadi_cataractlmm](2026_ahmadi_cataractlmm.md) | Cataract-LMM: Large-Scale, Multi-Source, Multi-Task Benchmark | 2026-05／arXiv 2025-10 | Scientific Data | 混合 | 汎化, ラベル・データセット | 2 施設 3,000 本のデータの元論文。別の施設で F1 が落ちる（MViT-B 77.1→57.6%）、凍結した DINO は落ちにくい（79.98→67.9%）。データは現在ダウンロード不可 | 未読 |
| [2026_chen_dsted](2026_chen_dsted.md) | DSTED: Decoupling Temporal Stabilization and Discriminative Enhancement for Surgical Workflow Recognition | 2026-??（IPCAI）／arXiv 2025-12 | IPCAI 2026 | 混合（推測） | 揺れ, 手法一般 | 確信度で長い文脈の重みを決めるゲート（2026-11-20 の「似て見える組を狙う長い窓」の比較相手）。**実験の結果しだいで優先度を上げる** | 未読（和訳あり） |
| [2026_guo_ttdf](2026_guo_ttdf.md) | TTDF | 2026-09（arXiv） | arXiv | 工学系 | 揺れ, 遅れ | 遅れを数字で出した唯一の論文（余計な切り替わりを減らすと遅れの中央値 0 → 6.3）。揺れと遅れのトレードオフの比較相手 | 未取得 |
| [2026_he_cauclip](2026_he_cauclip.md) | CauCLIP: Bridging the Sim-to-Real Gap in Surgical Video Understanding via Causality-Inspired VLM | 2026-02 | arXiv | 工学系 | 汎化, 手法一般 | CLIP＋周波数のデータ拡張＋因果でない特徴を抑える損失で、別の領域のデータなしに工程認識を頑健にする（VR→豚） | 未読 |
| [2026_liu_stabilizing](2026_liu_stabilizing.md) | Stabilizing Temporal Inference Dynamics for Online Surgical Phase Recognition | 2026-??（MICCAI）／arXiv 2026-05 | MICCAI 2026 | 未確認 | 揺れ, 評価方法 | 揺れの指標 TFI の出典。証拠を累積して切り替える方法。遅れは測っていない | 未取得 |
| [2026_mezzina_framework](2026_mezzina_framework.md) | A Framework for Real-Time Surgical Phase Recognition … | 2026-??（WACV） | WACV 2026 | 未確認 | 揺れ, 遅れ | 前向きだけの Viterbi で正確さと遅れを釣り合わせるらしい。遅れを扱う数少ない論文 | 未取得（403） |
| [2026_spencer_dataefficient](2026_spencer_dataefficient.md) | Data-Efficient Surgical Phase Segmentation in Small-Incision Cataract Surgery: A Controlled Study of Vision Foundation Models | 2026-04（arXiv） | arXiv | 工学系 | 手法一般, ラベル・データセット | 白内障で DINOv3 の特徴を比べた研究（7B が最良、L もほぼ同じ）。2026-11-20 の長い窓の枝を DINOv3 に替える根拠 | 未読（和訳あり） |
| [2026_tabuchi_four](2026_tabuchi_four.md) | Four learning phases in cataract surgery revealed by complication rates among novice surgeons | 2026-01 | Scientific Reports | 医者 | 臨床（医者の視点） | 新人の合併症率を平滑化＋変化点検出（PELT）で 4 段階に区切った。教育という用途の根拠。研修医の手術ほど合併症が多い | 未読（日本語要約あり） |
| [2026_tabuchi_grounded](2026_tabuchi_grounded.md) | Grounded ophthalmic AI beyond the electronic health record trap | 2026-10 | BMJ Health Care Inform | 医者 | 臨床（医者の視点）, 汎化 | 眼科 AI の設計の総説。施設ごとのワークフローへの適応を明示する層が要る、という医師側の主張 | 未読 |
| [2026_tabuchi_visual](2026_tabuchi_visual.md) | Visual alert integration with a time out monitor improves utilisation of an AI safety authentication system | 2026-10 | BMJ Open Quality | 医者 | 臨床（医者の視点） | 警告の見せ方（赤の点滅→緑）を変えるだけで AI が使われない割合が 66.8% 減った。リアルタイム警告の見せ方の参考 | 未読 |
| [2026_vijayasimha_comment](2026_vijayasimha_comment.md) | Comment on: Evaluating LLMs vs residents in cataract and refractive surgery | 2026-06 | J Cataract Refract Surg（コメント） | 未確認 | 臨床（医者の視点）, 評価方法 | 試験の正解率が高くても臨床で信頼できるとは限らないとして、LLM の検証の枠組みを提案。「正解率だけでは臨床の要件を測れない」という背景の補強 | 未取得（PDF なし） |
| [2026_xiang_distilled](2026_xiang_distilled.md) | Using distilled knowledge from large transformer models for efficient identification of cataract surgery steps | 2026-11 | Frontiers in Digital Health | 医者 | 手法一般, 臨床（医者の視点） | 眼科（Stanford）による知識蒸留で軽い手順認識（87.7%、計算量約 1/4）。研修医 268 本・6 施設。リアルタイム・手元で動かす用途 | 未読 |
| [2026_zhang_advances](2026_zhang_advances.md) | Advances in cataract surgery: is a new era on the horizon?（総説） | 2026-04 | Curr Opin Ophthalmol | 医者 | 臨床（医者の視点） | 眼科医の総説。AI の節で手術動画の工程認識（フレーム単位 90% 超）・器具の追跡・研修医の技能評価・合併症の早期予測を「来る時代の目標」と紹介。用途の根拠。施設間の性能や揺れには触れていない | 未読（日本語要約あり、原文 PDF 未取得） |
| [tbd_surgcbvr](tbd_surgcbvr.md) | SurgCBVR-CAT（クリップ検索ベンチマーク） | 未出版 | （Data in Brief 予定） | 未確認 | ラベル・データセット | Cataract-1K の工程ラベルを使った派生資源。論文が出たら登録し直す | 論文がまだない |
