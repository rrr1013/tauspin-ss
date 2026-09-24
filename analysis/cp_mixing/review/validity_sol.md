判定は **revise** です。密度行列、局所応答、標準的な \(\phi^*_{CP}\) 実装は概ね正しい一方、2つの中心主張が現在の証拠より強く書かれています。

## Findings

1. **重大 — 「truth入力の古典法より良い」は、truthの範囲が不完全で、完全truth比較では逆転する**

   - 何が問題か：`classical_truth`はtruth \(\pi^\pm,\pi^0\) を使いますが、ρの \(y\) はlab-frame energyのままです。[p4_classical_full.py:90](/Users/ryunosuke/Projects/tauspin/analysis/cp_mixing/p4_classical_full.py:90)、[p4_classical_full.py:98](/Users/ryunosuke/Projects/tauspin/analysis/cp_mixing/p4_classical_full.py:98)。
   - 数値：保存値ではρ×ρが古典truth `1.728°`、learned reco \(h\) `1.394°`です。[classical_full.json:95](/Users/ryunosuke/Projects/tauspin/analysis/cp_mixing/results/classical_full.json:95)、[classical_full.json:123](/Users/ryunosuke/Projects/tauspin/analysis/cp_mixing/results/classical_full.json:123)。
   - なぜ結論が変わるか：同じevent・同じtruth decay planeで、truth \(\nu\) からτ静止系へboostして標準の \(y_\tau\) を計算すると、独立再計算でρ×ρは `0.895°`、π×ρは `0.728°`となり、learned reco \(h\) の `1.394°`、`1.174°`を明確に上回りました。ρ×ρではlab/restの \(y_-y_+\) 符号が41.3%異なります。
   - 最小の確認：`pions_t`と`pi0_t`をtruth τ四元運動量の静止系へboostし、そのenergyから`y_t`を再計算する。
   - 境界：現実に再構成可能なLHC標準法がlab \(y\) を使うのは正しいです。文献もτ静止系 \(y_\tau\) を理想定義とした後、実装可能なlab \(y_L\)へ置換しています。[Berge–Bernreuther–Kirchner, Sec. II.B](https://arxiv.org/pdf/1510.03850)。したがって成立する主張は「full-reco learned \(h\) は、perfect visible pionを与えた標準lab-\(y\) \(\phi^*_{CP}\) より良い」です。「truth入力一般」や同一情報量での優位ではありません。

2. **重大 — Fisher情報は \(f\to0\) で対数発散しない**

   - 何が問題か：[p2_sensitivity.py:83](/Users/ryunosuke/Projects/tauspin/analysis/cp_mixing/p2_sensitivity.py:83)のtrimは、発散を正則化する操作として解釈されていますが、その前提が誤りです。
   - なぜ：\(f\to0\) では \(T_p=O(\sqrt f)\) も同時にゼロへ行くため、\(E_0[S^2]\) は有限です。無選別・\(\beta=1\) の解析値は
     \[
     I_\phi=E[S^2]=\frac43,
     \]
     すなわち10,000 eventで `0.4962°`です。独立MC積分もこの値へ収束しました。対数発散するのは \(E[S^4]\) であり、Fisherそのものではなく、その有限MC推定の分散です。
   - 結論への影響：`0.487°`を「発散するためtrim依存の参考範囲」と扱う説明は撤回すべきです。選択後の実測 `I=1.385` は有限で妥当ですが、通常bootstrapの誤差はheavy tailのため楽観的になり得ます。\(T_p\)による主結果は影響を受けません。
   - 最小の確認：等方unit-vector toyでsample sizeを増やして \(E[S^2]\to4/3\)、一方 \(E[S^4]\) が対数的に増えることを確認する。trim曲線は「情報量の大きいeventを捨てた場合の感度」と読み替える。

3. **中 — IPのCP/HZ「実効統計量」比はモデル依存**

   - 何が問題か：CP側は実際の局所感度比 \((\sigma_{\rm base}/\sigma)^2\) ですが、H/Z側はAUCを等分散Gaussianの \(d'=\sqrt2\Phi^{-1}(\mathrm{AUC})\) に変換しています。[p6_geometry_value.py:107](/Users/ryunosuke/Projects/tauspin/analysis/cp_mixing/p6_geometry_value.py:107)。この変換は一般のscore分布に対する実効luminosityではありません。またbaseはseed 43、IP群は主にseed 42です。[p6_geometry_value.py:31](/Users/ryunosuke/Projects/tauspin/analysis/cp_mixing/p6_geometry_value.py:31)。
   - なぜ結論が変わるか：「1.65対1.10、したがって増分で約4倍」は定量的には保証されません。
   - 最小の確認：同一seed ensembleで比較し、H/Z側もexpected log-likelihood/KLまたは固定検定の必要event数で換算する。
   - ただし：既存の同seed `base_s43 → full22_s43`を独立再計算するとCP `1.650`、H/Zの現行換算 `1.109`でした。weightを入れても方向は同じです。したがって「CPの方が強く改善する」という定性的結論は支持されます。

4. **中 — 「新しい学習なし」「最良readout」は言い過ぎ**

   - 何が問題か：[README.md:4](/Users/ryunosuke/Projects/tauspin/analysis/cp_mixing/README.md:4)に反して、`p5`はtrain splitのtruth scoreへ次数1–4のOLSをfitしています。[p5_readout.py:39](/Users/ryunosuke/Projects/tauspin/analysis/cp_mixing/p5_readout.py:39)、[p5_readout.py:60](/Users/ryunosuke/Projects/tauspin/analysis/cp_mixing/p5_readout.py:60)。これは小規模でも新しい教師あり学習です。
   - 数値：full22では三重積 `1.282°`に対し4次readout `1.227°`で、σが4.3%、実効event数が約9%改善しています。[readout.json:81](/Users/ryunosuke/Projects/tauspin/analysis/cp_mixing/results/readout.json:81)、[readout.json:139](/Users/ryunosuke/Projects/tauspin/analysis/cp_mixing/results/readout.json:139)。
   - なぜ結論が変わるか：低次数多項式だけから「最良readout」を確定できません。「三重積がかなり近い」は支持されますが、「最良とほぼ同じ」は未証明です。
   - 最小の確認：fitを新しい学習として明記し、train-onlyでより柔軟なconditional-score回帰を固定後、validationで三重積との差をpaired bootstrapする。

5. **中 — mode pair依存は弱いが、ゼロではない**

   - 何が問題か：[make_figures.py:216](/Users/ryunosuke/Projects/tauspin/analysis/cp_mixing/make_figures.py:216)の“does not care”は強すぎます。
   - 数値：exact \(T_p\)は `0.570–0.604°`（約6%幅）、optimalは `0.475–0.523°`（約10%幅）です。[sensitivity.json:348](/Users/ryunosuke/Projects/tauspin/analysis/cp_mixing/results/sensitivity.json:348)。
   - なぜ結論が変わるか：例えばπ×3πとρ×ρの \(T_p\) 差は掲載bootstrap誤差に対して約3σです。selectionでmodeごとのpolarimeter分布が変わるため、完全な普遍性は期待できません。
   - 最小の確認：「ほとんど依存しない」を「per-event感度のmode依存は6–10%に収まる」へ定量化する。

6. **軽微 — H/Z AUCの表示がweightedになっているが、入力値はunweighted**

   - 場所：[make_figures.py:209](/Users/ryunosuke/Projects/tauspin/analysis/cp_mixing/make_figures.py:209)。元のAUC実装はevent weightを受け取りません。
   - 影響：図3の軸名だけが不正確です。独立にweighted AUCを計算すると各modeで差は最大約0.002未満で、結論は変わりません。
   - 最小の確認：軸を`unweighted AUC`へ直すか、`overlap_weights`を用いたweighted AUCへ統一する。

7. **軽微 — `cp_density`はexactだが、scoreは有限βでexactではない**

   - 場所：[cp_tools.py:35](/Users/ryunosuke/Projects/tauspin/analysis/cp_mixing/cp_tools.py:35)。
   - なぜ：exactな横blockでは
     \[
     C'_{nr}(0)=2/\beta ,
     \]
     なので \(S=(2/\beta)T_p/f\) です。現コードは \(2T_p/f\)。
   - 影響：このsampleでは \(\beta=0.999240\)、補正は0.076%だけで全結論に無視できます。ただし「exact in beta」とscoreの記述は一致していません。
   - 最小の確認：`c_matrix(phi)`の数値微分をscoreへ使うか、明示的に`2/beta`を掛ける。

## 確認した項目

- CP密度行列：確認した。projector、τ⁺の \((\not p-m)\)、spin 4-vector、\(\bar\Gamma=\Gamma\)、B/C差分抽出はいずれも正しい。有限βで
  \[
  C_T=\frac1{\beta^2\cos^2\phi+\sin^2\phi}
  \begin{pmatrix}
  \beta^2\cos^2\phi-\sin^2\phi & 2\beta\sin\phi\cos\phi\\
  -2\beta\sin\phi\cos\phi & \beta^2\cos^2\phi-\sin^2\phi
  \end{pmatrix},
  \quad C_{kk}=-1 .
  \]
  URの \(R(2\phi)\) との差の最大値は `7.60e-4`。Bはゼロです。文献の式とも一致します。[Berge–Bernreuther–Kirchner, Eq. (9)](https://arxiv.org/pdf/1510.03850)。

- 規約：確認した。side 0は全eventで総電荷−1、side 1は+1。`k=τ+`、`n=beam×k`、`n×r=k`で、文献の`k=τ−`表記を変換すると同じ \(+2T_p\) 符号になります。canonical \(h=-h_{\rm physical}\) の両側反転はbilinearとtriple productの双方を変えません。

- 応答式：確認した。selected populationではscoreから平均を引くnormalization項が必要ですが、`Cov(T,S)`と`Var(S)`は定数に不変なので現実装でよいです。shape-only・固定selected-event数の結果であり、rate/acceptance情報は含みません。

- reco \(T\) とtruth \(S\)：確認した。detector responseがφに依存しない限り、complete-data scoreとの共分散でreco observableの応答を評価できます。truth-score CRはexact-\(h\)上限であってreco-level CRではありません。

- 古典法の幾何：確認した。荷電π対ZMF、横成分、\(\hat q_-\cdot(\hat n_+\times\hat n_-)\)、\(y\)符号反転、IPを\((0,\hat n)\)としてboostする処理は標準定義と一致します。意図的に符号やplaneを弱くした形跡はありません。

- 数値整合：確認した。全validation artifactで`global_indices/event_numbers/labels/modes`が一致し、IP配列も`event_numbers[global_indices]`が一致しました。train/validationの`global_indices`および`(label,file,entry)`重複はゼロ。H/Zの取り違えもありません。

- shuffle：確認した。eventと`h_pred`の対応を壊すnullとして正しく、応答はゼロと整合します。ただしlabel shortcutやdetector systematicを否定するcontrolではありません。

レビュー対象はcommit `581def5`のp0–p7と保存結果です。レビュー中に別プロセスから`make_figures.py`、`fig5`、新規p8への未commit変更が現れましたが、それらは対象に含めず、私はファイルを変更していません。