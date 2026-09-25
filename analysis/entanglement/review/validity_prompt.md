あなたは独立のvalidity reviewerです。読み取りのみ行い、ファイルを変更しないでください。

# 対象

`~/Projects/tauspin/analysis/entanglement/`（branch `ariadne/auto-2026-09-26-entanglement`）。
H→ττ / Z→ττ の ditau spin密度行列を再構成polarimeter `h` から推定し、量子もつれを判定できるかを
測った探索runです。新しい学習も新しいMC生成もしていません。既存artifactをnumpyで評価しただけです。

`README.md`に全体像、`ent_tools.py`, `z_density.py`, `ent_data.py`, `q0`〜`q4` にコード、
`results/*.json` に数値、`figures/*.png` に図があります。入力npzは `../cp_mixing/data/` にあり、
`export PYTHONPATH=../mode_pair_auc_origin:../cp_mixing` を付ければそのまま再現できます（numpyのみ）。
基底と `h` の規約は `../mode_pair_auc_origin/polarimeter.py` の `frames()`、
Higgs側のspin密度は `../cp_mixing/cp_density.py` が正本です。

# 今回答えてほしい問い

1. **生成状態**。`cp_density.py`（H）と `z_density.py`（非偏極Z）が返す \(B_\mp, C\) は正しいか。
   - H が \(\beta\) に依らず \(C=\mathrm{diag}(1,1,-1)\)、すなわち飛行軸まわりの \(m=0\) triplet
     （純粋Bell状態）になる、という主張は正しいか。CP-oddなら singlet になるという対比も含めて。
   - `z_density.py` の Z polarization sum \(-g^{\mu\nu}+q^\mu q^\nu/M^2\) の実装、
     \(\bar V = \gamma^0 V^\dagger \gamma^0\) の評価、metric の上げ下げは正しいか。
     出てくる \(C_Z=\mathrm{diag}(0,0,+1)\)、\(B_\mp=+0.147\,\hat k\) は
     非偏極 Z→ττ として文献と整合するか（\(P_\tau=-2v_\tau a_\tau/(v_\tau^2+a_\tau^2)\) との符号関係を含む）。
   - 「v2のZは pp→Z+j と Z→ττ を別生成したので Z は非偏極」という前提のもとで、
     この \(C_Z\) を使ってよいか。beam軸に依存しないという議論は正しいか。

2. **entanglement observable**。`ent_tools.py` の `theta_to_density`, `partial_transpose`,
   `concurrence`, `observables`, `witness_coefficients`, `project_physical` は正しいか。
   - canonical `h` が物理polarimeterの \(-1\) 倍であることから単一spin blockだけ符号反転する、
     という扱いは正しいか。\(C\) が不変という主張は正しいか。
   - `witness_coefficients` が返す線形汎関数は本当に
     「\(\rho_H\) で \(-1/2\)、あらゆる分離可能状態で \(\ge0\)」という性質を持つか。
   - Horodecki \(m_{12}=s_1^2+s_2^2>1\) を「CHSH破れが可能」の条件として使うのは妥当か。
     \(B\ne0\) の混合状態（Z）にも適用してよいか。

3. **推定量**。`ent_tools.calibrate` / `unfold` の
   \(E_\theta[g]=(A+R\theta)/Z(\theta)\)、\(Z(\theta)=1+E_0[\phi]\cdot\theta\)、
   \((R-\langle g\rangle e^{\mathsf T})\hat\theta=\langle g\rangle-A\) という導出は正しいか。
   - 「校正sampleと測定sampleが同じ分布なら、推定量は仮定した状態をそのまま返す」という
     識別可能性の主張（run noteの第3節）は正しいか。反例はあるか。
   - その結論を受けてAsimov形式（中心値＝生成状態、行ごとに違うのは \(\sigma\)）で報告しているが、
     この報告形式は正当か。\(\sigma\) の意味（校正を固定し測定eventのみbootstrap）は妥当か。
     delta法との一致（`witness_sd_delta_method`）で十分か。
   - \(1/f\) 重みは \(f=1+h_-^TCh_+\in[0,2]\) なので \(f\to0\) で重い尾を持つ。
     ESS 0.41 という条件下で \(E_0\) の推定は信用できるか。\(E[1/f^2]\) の発散は問題か。
   - `moment_unfold_C`（観測2次momentで割り戻すだけ）の残差bias 0.026 は何に由来するか。
     この推定量の系統を見積もる方法はあるか。

4. **装置的相関の主張**。「非偏極測度で \(E_0[h_{\rm pred,-}\otimes h_{\rm pred,+}]\) が
   \(0.037\) あり、生の横相関の55%を占める」（run note第5節、`fig3`）は正しい読みか。
   - これは本当にspin非依存か。それとも \(1/f\) 重みの残差や推定の癖ではないか。
   - 「geometryを足すと装置的割合が 0.64→0.55→0.40 に下がる＝IP/SVが共通modeを本物のspin情報へ
     置き換えている」という解釈に、別の説明は残っていないか。
   - 縦成分 \(kk\) にこの効果がほとんど無いことは、何を意味するか。

5. **Z対照**。「H eventsで校正した応答をZ eventsへ当てても \(\langle W\rangle\) は正のまま」
   （run note第6節）は、「再構成がもつれを製造していない」の証拠として十分か。
   - 返る \(C_{kk}=1.13>1\) は物理領域外で、その結果 \(\hat\rho^{T_2}\) の最小固有値が負になる。
     「PPT固有値が負」より「事前固定した線形witness」のほうが安全、という結論は正しいか。
   - cohortをまたぐ応答の移送のbias \(|\Delta C|\simeq0.1\)〜\(0.2\) は、何が原因か。

6. **到達点の数字**。「もつれ判定に要る信号event数 46（exact h）→143（ideal IP）→240（+IP/SV）
   →344（geometryなし）」は、この定義のもとで正しく計算されているか。
   - \(N(3\sigma)=N_{\rm obs}(3\sigma_W/|\langle W\rangle|)^2\) というscalingは妥当か。
     \(\langle W\rangle\) は有界な演算子の期待値なので、少数eventでGauss近似は成り立つか。
   - \(m_{12}>1\) の到達点がもつれの約9倍になるのは妥当か。
   - 「信号eventのみ、背景・系統なし」という但し書きで十分か。他に落としている前提はあるか。

7. **mode pair**。ρ×ρのreco劣化が最悪（7.7倍）でπ×πのexact `h` 感度が最悪（86 events）という
   結果に、統計的に無理のない説明が付いているか。π×πの説明（\(p_0\) の非等方性）は検証されているか。

8. **図**。`figures/*.png` を実際に開いて見てください。探索段階の図として
   （最終claim用のfigure setとしてではなく）、軸名、単位、population、selection、normalization、
   binning、legend、色だけに頼らない区別、誤解を生む表示が無いかを見てください。
   中心の主張を確かめるのに必要なのに**作られていないview**があれば指摘してください。

9. 上記以外で、この run の結論を弱める／覆す実装上の誤り、統計の誤り、物理の誤りがあれば指摘してください。

# 出してほしい形

finding ごとに、(a) 何が問題か、(b) それが結論のどれをどう変えるか、(c) 確かめ方または直し方、
(d) 深刻度（結論を覆す／数値を変える／表現の問題）。正しいと確認できた点も明示してください。
根拠のない同意はしないでください。
