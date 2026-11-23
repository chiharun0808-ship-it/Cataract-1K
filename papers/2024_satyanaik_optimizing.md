# Optimizing latent graph representations of surgical scenes for unseen domain generalization

- 引用キー：2024_satyanaik_optimizing
- 著者：Satyanaik S, Murali A（同等貢献）, Alapatt D, Wang X, Mascagni P, Padoy N（ICube／Strasbourg 大・IHU Strasbourg、Mascagni は Fondazione Policlinico Gemelli も）。**混合**
- 発表：2024-04（Int J CARS 19:1243–1250、オンライン 2024-04-28）
- リンク：https://doi.org/10.1007/s11548-024-03121-2
- ライセンス：未確認（PDF に「© CARS 2024」とあり、オープンアクセスの記載なし。PDF は出版社のサイトから取得できた）→ 全文和訳はしない
- 状態：未読（2026-11-24 登録）

## 一言で
手術画面を物体（器具・解剖）のグラフとして表す object-centric な方法が、**学習に使っていない施設（unseen domain）**でどれだけ通用するかを比べ、分離を促す損失を足した LG-DG で最良の比較手法より 9.28% 改善。

## 何をしたか
- 題材は腹腔鏡下胆嚢摘出術の CVS（critical view of safety）判定。白内障ではない。
- 施設：Endoscapes2023（フランス Strasbourg）と、新しく作った Endoscapes-WC70（中国 四川、70 本）。
- 施設間のずれの原因として「手術の流れ、カメラの設定、患者の層」を挙げている（要旨）。

## 自分の研究との関係
- 汎化（他施設で使えるか）の手法の例。物体（器具・解剖）に分けて表すと施設が変わっても強い、という主張は、器具の先端の周りを見る案（tip-attention）と考え方が近い。
- 違うところ：工程認識ではなく静止画の判定、白内障ではない。
