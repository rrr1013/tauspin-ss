あなたは独立のvalidity reviewerです。読み取りのみ行い、ファイルを変更しないでください。

# 対象

`~/Projects/tauspin/analysis/cp_mixing/`（branch `ariadne/auto-2026-09-25-cp-mixing`）。
Hgigs→ττのspin相関から、τ Yukawa結合のCP mixing angle φ_τ への感度を測った探索runです。
新しい学習・新しいMC生成はしていません。既存artifactをnumpyで評価しただけです。

`README.md`に全体像、`cp_density.py`, `cp_tools.py`, `p0`〜`p7`, `make_figures.py`にコード、
`results/*.json`に数値、`figures/*.png`に図があります。入力の所在も`README.md`にあります。
（大きい入力npzは`data/`にあり、そのまま実行して再現できます。numpyのみ必要です。）

# 今回答えてほしい問い

1. **CP密度行列**。`cp_density.py`が計算している \(C(\phi_\tau)\) は正しいか。
   spin projector \((1+\gamma_5 \not s)/2\) の使い方、v-spinor側（τ⁺）の扱い、spin 4-vectorの構成、
   \(\bar\Gamma = \gamma^0\Gamma^\dagger\gamma^0\) の評価、6方向差分によるB・C抽出は妥当か。
   結果（φ=0でdiag(1,1,−1)、φ=90°でdiag(−1,−1,−1)、中間で横blockがR(2φ)、B=0、β補正<1e-3）は
   文献のHiggs CP混合のττ spin相関と整合するか。符号・handednessの規約が
   `analysis/mode_pair_auc_origin/polarimeter.py`の`frames()`（k=τ⁺方向、n=beam×k、右手系、
   canonical h = −物理h）と一貫しているか。

2. **推定量**。`cp_tools.py`の
   score \(S = 2 T_p/f\)、\(T_p=(h_-\times h_+)\cdot\hat k\)、\(f = 1+h_-^{T}C_0h_+\)、
   線形応答 \(\mathrm{d}\langle T\rangle/\mathrm{d}\phi|_0 = \mathrm{Cov}_0(T,S)\)、
   \(\sigma_N = \sqrt{\mathrm{Var}(T)/N}/|\mathrm{Cov}(T,S)|\)、
   Cramér–Rao \(1/\sqrt{N\,\mathrm{Var}(S)}\) は、この設定で正しいか。
   normalizationの微分項を落としてよいか。selection（可視pT>20 GeVなど）が入った母集団でも成立するか。
   Fisher情報が \(f\to0\) で対数発散することの扱い（`p2`のtrim安定性）は適切か。
   observable T がreco量、score S がtruth量であることは、この推定量を壊さないか。

3. **古典 φ*_CP**。`p3_classical.py`/`p4_classical_full.py`の実装は、
   標準のimpact-parameter法・neutral-pion(decay-plane)法として妥当か。
   荷電π対の静止系へのboost、解析ベクトルの横成分、符号（\(\hat q\cdot(\hat n_+\times\hat n_-)\)）、
   ρ側の \(y\) 符号反転、π側IPの4元ベクトル化(0, n̂) の扱いに誤りはないか。
   古典法をわざと弱く実装していないか（fairnessの点検）。

4. **数値結果の妥当性**。`results/`の数値に、内部矛盾や取り違えがないか。
   特に、行の整列、H/Zの取り違え、train/validationの混入、shuffle controlの解釈。

# 主張しようとしていること（これを支持するよう誘導しないでください）

- exact `h` での φ_τ 感度は崩壊mode pairにほとんど依存しない。
- reco段階では、学習した `h` が古典 φ*_CP より良い。ρ×ρでは、truth入力を与えた古典法よりも良い。
- impact parameterはH/Z判別よりCP角に対して実効統計量で大きく効く。
- `h_pred` からの最良readoutは、その三重積とほぼ同じ。

これらのうち、証拠が足りないもの、別の説明が残るもの、実装の誤りで説明できるものを指摘してください。

# 出力

findingを重要度順に。各findingは「何が問題か／どのfile:行または数値か／なぜそれが結論を変えるか／確かめる最小の方法」。
問題が無い項目は短く「確認した」とだけ書いてください。ファイルは変更しないでください。
