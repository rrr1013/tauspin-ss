# HH→bbττ：TauSpin・lepton寿命情報の懐疑的レビュー

レビュー日：2026-10-08。対象は `../review_packet.md`、親ディレクトリのPython実装8本、`summary.md`、主要図7枚。図はすべて画像として開いて確認した。コードの実行による再解析、未提供のprepared cohort・remote workspaceの検証は行っていない。以下の数値は提供表の監査であり、独立再現値ではない。本レポート以外のファイルは変更していない。

**版の注意：** 最終照合で`lifetime.py`、`calibrate.py`、`projection.py`の同時更新を検出した。更新箇所も読み、コードの参照行は最終確認した版へ合わせた。packet・summary・図は開始時と同一だった。数値の判定はこの提示packetに対するものであり、追加された設定での再計算値は未提供。コードと結果の整合性についてM7に記載する。

**結論：寿命情報に識別力がある物理的根拠はある。しかし、提示された増分をATLAS HL-LHCの感度向上として扱うには、既存選択後の残余情報、実際の最終分類器との相関、背景のlepton起源を検証する必要がある。C1は探索的モデルの結果へ文言修正すれば可。C2のATLAS感度・頑健性の主張と、C3の上限の主張は現状不可。**

重大度の意味：blockingは中心的な実験感度の主張を妨げる未検証事項、majorは結果または解釈を大きく変え得る事項、minorは表示・数値監査上の修正事項を指す。「過大になり得る」と「過大であると実証された」は区別する。

## 数値の読み方

「lepton寿命で+4%」は、legacyの **bbττ全体のZ比** `R=1.0410` を指す。τℓτh単独は `R=1.1492`、HH全チャンネル結合の換算値は4.367σで **+2.51%**。最新Run 3組成ではそれぞれ1.0483、1.1483、4.386σで、結合の増分は+2.96%。spinと寿命を合わせた結合換算4.420–4.425σの+3.76–3.87%とは異なる。根拠：`summary.md:17–20,29–31`、`../review_packet.md:28–32`。

| 情報 | legacyのR(bbττ) | latest Run 3のR(bbττ) | packetの換算Z(HH結合) |
|---|---:|---:|---:|
| TauSpin spinのみ | 1.0147 | 1.0120 | 4.298 / 4.291σ |
| lepton寿命のみ | 1.0410 | 1.0483 | 4.367 / 4.386σ |
| TauSpin spin＋寿命 | 1.0610 | 1.0630 | 4.420 / 4.425σ |
| exact h＋寿命 | 1.1199 | 1.1168 | 4.577 / 4.569σ |

根拠：`summary.md:17–23,29–31,47`。右列は公式workspaceの再fit結果ではなく、後述の換算式の出力。

## Findings：blocking

### B1 — 既存lepton ID・TTVAの後に残る寿命情報を測っていない

**根拠。** 2404.12660 §4は物体再構成・識別を2209.10910から継承する。2209.10910 §5.1・Table 2では、signal electronはTight、muonはMediumを要求する。そのelectron物体定義の参照先1908.00005は、Table 1（p.15）と§6.1（p.26）で **d0とその有意度をelectron likelihood IDに使う**ことを明記する。したがって、独立のd0 cutがないことは、IDに寿命情報が入っていないことを意味しない。[2404.12660 §4](https://arxiv.org/html/2404.12660v2)、[2209.10910 §4・§5.1/Table 2](https://arxiv.org/html/2209.10910)、[electron性能論文 Table 1・§6.1](https://arxiv.org/pdf/1908.00005)。

packetの寿命試料はparametricなSLT-like選択であり、提供コードは外部の`prepared.npz`にある選択flag・d0列を読む。ATLASのIDを通した条件付き分布になっているか、electron IDでIP入力を外した特別設定があるかは提供物から確認できない。根拠：`../lifetime.py:4–10,25,77–87`、`../review_packet.md:21`。図3の大きなτ→ℓ tailは、ATLASの選択後にも同量残ると確認されたtailではない。

**TTVAの確定範囲。** 旧解析に標準の独立cutが実際にあるかは、下の参照追跡結果のとおり未確定。最新2607.26879 §4はτのleptonic decayの受容を保つため横impact-parameter選択を課さない旨を明記し、e/μには `|z0 sinθ|<0.5 mm` を記載する。この最新の記述を2024年のprojectionへ遡及適用できない。[2607.26879 §4](https://arxiv.org/html/2607.26879)。

**二重計上の判定。** 確定した「二重計上」とは断言しない。IDで同じ変数を使っても、選択後に残るd0分布には追加情報があり得る。問題は、その残余情報を測らず、別の選択・分解能の分布をpost-fit yieldへ接続している点である。これにより増分が過大になる可能性がある〔推測〕。

**対処・反証。** 実際のelectron ID設定、muon選択、TTVA、isolation、triggerを固定し、eとμを分けて「既存選択＋既存分類器」と「同じ選択・分類器＋d0」を比較する。選択後のpT・η・σ(d0)を揃えた条件付き比較で増分が消えるなら、独立した寿命情報の追加という解釈は反証される。TTVAを緩める場合は新しい解析として、signal受容とnonprompt背景も再評価する必要がある。

同時更新でelectron-only 5σ cutの`ecut5`が追加された（`../lifetime.py:32–36`）。これは説明にもあるとおりproxyであり、他のID入力との相関を含むelectron likelihoodへの条件付けを再現した証拠ではない。

### B2 — d0とspinの積、および代理Kへの対応付けは、既存スコアへの追加効果を保証しない

**根拠。** `../projection.py:201–205` は明示的に

`p(Dspin, d0sig | process, base bin) = p(Dspin | process, base bin) × p(d0sig | process, base bin)`

という積を作る。さらに、base binとの対応は自作Kの **signal累積分位** を合わせるものに限る（`../lifetime.py:90–102`、`../projection.py:112–136`）。ATLAS bin内の背景pT・η・起源・ID応答が一致する条件ではない。τℓτhのspinテンプレートはmass categoryだけで選び、K window条件も適用していない（`../projection.py:138–142,233–235`）。VBFの寿命テンプレートはHi/Loに対応しないinclusive選択になる（`../lifetime.py:72–74,91`）。

d0はleptonのpT、decay geometry、分解能に依存する。spinはτ decayのエネルギー・角度に依存する。tt̄のdirect-W、τ-mediated、heavy-flavour起源も既存kinematics・isolation・spin応答と関連する。**processでだけ条件付けてoriginを混合した積**では、originとspinの相関も失われる。背景の運動学差を新しい寿命情報として数えている可能性、同じdecay情報をspinと寿命の二つの独立測定として強めている可能性がある〔いずれも推測。バイアスの符号は未確定〕。

2209.10910 Table 3、2404.12660 §4.3にはlight-lepton d0を明示した最終分類器入力は見当たらない。しかし、入力のkinematicsがd0と相関する可能性は残る。最新2607.26879 §7の高度な分類器へ自作Kの分位を移すだけでは、その追加効果を検証できない。[2209.10910 §5.2/Table 3](https://arxiv.org/html/2209.10910)、[2404.12660 §4.3](https://arxiv.org/html/2404.12660v2)、[2607.26879 §7](https://arxiv.org/html/2607.26879)。

**対処・反証。** 同じevent群から `p(Dspin,d0sig | process, origin, actual baseline score bin)` を作り、実測joint版とfactorised版のRを比較する。少なくともK・pT・η・originを細分してclosureを調べる。条件付きshuffleで積近似に戻した場合だけgainが増えるなら、独立仮定による過大評価が示される。Kの`tokens/globals`のfeature定義も提示する必要がある（`../lifetime.py:39–40`）；提供コードだけでは「kinematic-only」を監査できない。

### B3 — 公式ATLAS感度への換算は、profile likelihoodの応答として未検証

**根拠。** packetは別年代のpost-fit yieldとsurrogate nuisanceからRを求め、`Zbbττ=3.54R`、一定の `c=0.880` で結合値を換算する（`../review_packet.md:15–19,22`、`../study.py:47–55`）。cは公開値のquadrature sumと結合値の比であり、bbττを改善したときの結合likelihoodの微分、nuisance covariance、CRによる拘束を与えない。

公式projectionは2404.12660を元に、**pre-fit templateを用い、Z+HFのみnormalisation×1.3を適用**する。全processのpost-fit shape・yieldを使うこととは同値ではない。公式結合は共通nuisanceを含むlikelihoodで実施する。[ATL-PHYS-PUB-2024-016 §2–3、Table 4](https://inspirehep.net/files/735311e4e52d37c9117fc7cde5b69aff)、[ATL-PHYS-PUB-2025-006 §3のbbττ、§4/Table 5](https://inspirehep.net/files/5cabc0e66dede649b8f6e37147e87a82)。

自身のcalibrationも、systematics/stat比がcomb/had-had/lep-hadで0.74/0.80/0.87、公式は約0.76/0.775/0.783である（`../review_packet.md:19`、図7右）。寿命のgainが生じるlep-hadの比は約11%相対的に高い。またcalibrationのlep-had値はSLT/LTTの別fitをquadratureでまとめる実装で、studyの共有nuisanceを含むlep-had joint fitと同じではない（`../calibrate.py:24–29`、`../study.py:35–43`）。このcalibrationから追加情報の効果まで正しく再現されるとは言えない。

同時更新はtop/fakeのbase-scoreに線形tiltを与えるnuisanceと二次元calibration scanを追加した（現行`../projection.py:265–272`、`../calibrate.py:36–49`）。この改善用設定の結果は提示表・図へまだ反映されておらず、各bin内のd0 shape/origin nuisanceでもない。仮にbaseline degradationを揃えてもB1・B2と換算式の問題は残る。

**対処・反証。** 4.42σ等には「公開値を使った経験的換算」と明示する。ATLAS感度の主張には、対応するbaselineのSR/CR・shape nuisanceを再現し、その同じlikelihoodへ追加情報を入れる必要がある。異なるnuisance構成でbaseline Zが同じでもRが変わるなら、一点calibrationから増分を移植する解釈は反証される。legacyとlatestのRの一致は、同じ仮定を共有する二つの計算の一致であり、独立した実験検証ではない。

## Findings：major

### M1 — 背景のτ→ℓ割合・nonpromptモデルにgainが強く依存する

**根拠。** 提示packetは全bin共通でtopのτ→ℓ割合0.08、single-H 0.92、fakeのnonprompt割合0、otherをpromptに固定する（`../review_packet.md:21`）。現行`../projection.py:186–198`もこの基本設定を持つが、同時更新でfakeにもτ→ℓ割合を追加した（M7）。`../lifetime.py:115–118` が求めるbin別fractionはlikelihoodに使用しない。背景のτ→ℓ形状もprocess固有ではなくHH signal形状を共用し、prompt/nonpromptは全cohortでpoolする（`../projection.py:165–183`）。nonpromptは平均100 µmの指数変位を横方向に射影したsynthetic stressで、heavy-flavour decay chain・選択との相関を実証したモデルではない（`../lifetime.py:12–14,81–84`）。

図3右のtop fraction 0.05–0.09は、MCが50 events以上ある低〜中スコアbinでの表示である。高スコア側のtop/fake点が欠けており、全15 binsの確認を意味しない（`../make_figures.py:133–136`）。図6では追加情報の寄与が高スコアbinへ集中する。したがってgainを決める領域で0.08が確認された、という読み方は不可。

| legacy scenario | 寿命のみR(bbττ) | spin＋寿命R(bbττ) | spin＋寿命の結合換算 |
|---|---:|---:|---:|
| nominal | 1.0410 | 1.0610 | 4.420σ |
| 分解能degraded | 1.0249 | 1.0425 | 4.371σ |
| top τ→ℓ 25% | 1.0232 | 1.0446 | 4.376σ |
| fake nonprompt 30% | 1.0217 | 1.0356 | 4.353σ |
| TTVA形状cut | 1.0170 | 1.0339 | 4.348σ |
| 上記同時＋spin contrast nuisance | 1.0062 | 1.0197 | 4.311σ |

根拠：`summary.md:27–42`、図5。寿命単独の増分は4.1%から0.62%まで変わり、nominalから約85%失われる。これは依存性を示す。選んだscenario内で正の値が残ることは、4.31–4.38σが信頼区間・物理的下限であることを示さない。

**対処・反証。** sensitive binのtopをdirect-W、τ-mediated、heavy-flavour/nonpromptに分け、single-Hをproduction/decay mode別に確認する。W+jetsは専用テンプレートとして独立に監査されておらず、fake/otherの仮定で包含したことを明記する。fractionとtail shapeを測定に基づくnuisanceとしてprofileし、signalの寿命形状をbackgroundへ共用した場合とのR差を示す。high-score topのτ成分が大きい、またはfakeの変位tailが大きいと確認され、gainが消えるなら、prompt除去によるnominal増分は反証される。

### M2 — TTVA variantは受容変化を含む再解析ではなく、正規化した形状の切り詰め

**根拠。** `../lifetime.py:104–114` は3σ/5σ cut後のhistogramをunit-normaliseする（`hist`:59–61）。`../projection.py:238–239` はそれに元の公開yieldを掛ける。cutを通るsignal/background効率はyieldへ反映しない。studyのbaselineもlifetimeを設定する前に計算する（`../study.py:69–70`）。

**影響。** `lt_ttva`のR=1.0170/1.0339は「cutでtailを除いた後も同じ総yieldがある場合」の結果である。標準TTVAを新たに課した解析の感度や、既存TTVAを緩めたときの改善には対応しない。既存yieldが既に同じcut後のものなら、形状側の条件もそれに整合させる必要がある。

**対処・反証。** 既存選択を維持する比較と、cut変更で受容を変える比較を明確にする。後者では各processのcut効率とorigin mixtureを更新する。現状の表記は「TTVA相当の形状切り詰めstress」へ変更する。

### M3 — ITkの公開性能と提示smearingは桁が近いが、lepton d0 significanceモデルとして未校正

**根拠。** packetの `σ=√(10²+(150/pT)²) µm`、electron×1.2、2%の4σ-width tailは、提供実装では外部prepared dataの説明として記載されるだけである（`../lifetime.py:4–7,80`、`../review_packet.md:21`）。公開ITk資料との比較は後節に示す。PU200の公開offline reference曲線とは桁が近く、式が一律に楽観的と断定する証拠はない。しかしpTだけの式ではη、粒子種、core幅とtail、PV誤差を固定できない。electron×1.2と2%/4σの数値組に対応する一次根拠は確認できなかった。

trackのreco−truth residual幅と、再構成PVから測るprompt-lepton d0の幅は異なる。PV基準なら、独立近似で `σ²(d0 relative PV)≈σ²(track)+uᵀV_PV u` が必要で、track/PVの相関があれば共分散項も必要である〔測定上の推論〕。beamline基準の標準ATLAS有意度はbeam-widthを含む定義であり、packetのPV基準へ数値をそのまま移せない。[Muon性能論文2012.00578 §5.2](https://arxiv.org/html/2012.00578)。

**対処・反証。** d0の基準点とσの定義、PVへのtrack inclusion、PV association、e/μ別pT×η residual/pullを固定する。Z→ℓℓでprompt tailを校正し、同じID後のτ→ℓ試料で確認する。正しいpull・tailを入れた後に増分が消えるなら、名目gainは寿命より分解能の仮定に起因する。今の許容表現は「公開性能を参考にした探索的smearing仮定」。

### M4 — 有限テンプレート統計と寿命shape不確かさをfitに伝播していない

**根拠。** spin histogramの誤差を計算しても`T, _`として捨てる（`../projection.py:233–235`）。signal leptonが200未満のbinは、よりsignal-likeな隣接binの寿命形状を借りる（179–180）。nominalのmodifierは主にnormalisationで、spin contrastはoption、寿命shape/origin fraction/有限MC統計のmodifierはない（242–290、`DEFAULT`:373–374）。

cross-fittingとearly stoppingは学習上の確認として適切だが、tail histogramの統計誤差を消さない。`seed sd`は同じevent群での学習seed差であり、template統計、背景起源、detector modellingの誤差ではない。`none`と`exact`は1 seedなので0を出力する実装である（`../study.py:73,83`）。公式projectionの将来MC統計を無視する仮定も、今回の有限cohortから推定した形状が既知になることを保証しない。

**対処・反証。** 同じeventの再利用を保ったbootstrapまたは独立sampleでRの安定性を確認し、weighted effective countsとtail bin countsを示す。joint tailの有限統計と寿命形状をprofileする。隣接bin借用とshape uncertaintyを入れたときgainが大きく減るなら、細分化したbinの既知形状という仮定が増分を支えていたことになる。

### M5 — 「exact h」は現在のtoyのoracle比較で、C3の情報上限ではない

**根拠。** exact armは同じHH kinをphysical spin densityでreweightした解析posteriorで、lep-hadは **hadronic側だけ** を使う（`../spin_response.py:7–23,143–162`）。leptonic polarimeterを含む完全なτℓτhのjoint情報ではなく、寿命測定もnominalな有限分解能のまま。6 sub-binsへの圧縮とspin×寿命の積を通した数値であり、完全なjoint likelihoodの情報量ではない。

bin依存も残る：spinのみのexactはnsub4/6/10でR=1.0620/1.0750/1.0725（`summary.md:48,51,54`）。finite-bin oracleが真のreco armを数学的に包摂していること、すべての入力がexact hのみによる同じ応答channelから生成されることも証明されていない。従って4.57σを厳密な上限と呼べない。上限がそれより上か下かも、現在の実験モデルでは未確定。

**対処・反証。** 表記を「固定したspin-density・背景・分解能・binning仮定でのexact-h oracle benchmark」とする。上限を主張するなら、対象情報の集合、response、完全なjoint densityとnuisance条件を定義し、そのモデル内で上限性とbinning収束を示す必要がある。

### M6 — 真のτのreweightingはfake-τ応答を検証しない

**根拠。** 全spin仮説はspin-flat HHの同じ真のτ eventから作る。UやWUはunpolarised τまたはW τとのspin densityであり、jet→τやlepton→τの実際のdetector responseではない（`../spin_response.py:3–17`、`../projection.py:34–36`）。lep-hadのhadronic側テンプレートもhad-had試料の両側をstackして作る（`../spin_response.py:152–160`）。fixed-readout H/Z AUC一致だけでは、top・fake・single-Hの実際の条件付き応答まで検証されない。

spin-onlyのTauSpinはfake仮定でR=1.0102–1.0186、exactは1.0382–1.0779へ変わる（`summary.md:7–11`）。図1の高スコア領域にfakeが相当量あるため、C1の因果説明・C3の数値に影響する。single-HがHHと同じspin densityで区別できないという説明はこの固定kinモデル内では整合するが、「fakeではspin情報がない」と一般化する根拠にはならない。

**対処・反証。** process別・実際のτ ID後のresponseを使うか、現在の値をfake-response scenarioに条件付けた結果と明記する。data fake control sampleで高スコア側のspin応答がH-likeなら、現在のU/WUから得た追加効果は成立しない。

### M7 — 提示数値と現行コードのorigin仮定が一致しない

**根拠。** packetではfakeのlight-leptonをpromptとするが、更新後の`../projection.py:192–198`は`fake_tau_fraction`をdefaultでtopと同じ0.08とし、nonprompt以外をτ/promptへ分ける。更新後の`../calibrate.py:48–49`は新しい`calibration_v2.json`を出力する一方、提供図7は従来のcalibrationを示す。開始時とのhash比較でこのコード変更を確認したが、packet・summary・7枚の図に変更はなかった。

**影響・対処。** これは修正の意図を否定する指摘ではない。しかし、現在のコードを実行すれば提示表が再現されるとは言えない。結果に使ったcode/data/configの版を固定し、fake起源、electron ID proxy、shape設定を明示した再計算後に表・図・claimsを揃える。新しい設定の改善量は本レビューでは推測しない。

## Findings：minor

### m1 — 「+8% equivalent luminosity」は統計律に限定した換算

`(4.42/4.26)²−1=7.65%` は `Z∝√L` を仮定した計算である。baseline systematics込みの公式Table 5は2 ab⁻¹で3.71σ、3 ab⁻¹で4.26σであり、この二点の有効指数は `log(4.26/3.71)/log(1.5)=0.341`。単純な√L則ではない。[PUB-2025-006 Table 5](https://inspirehep.net/files/5cabc0e66dede649b8f6e37147e87a82)。**対処：**「√L則で形式的に換算すると約8%」と限定するか削除する。二点から3 ab⁻¹超へ厳密な等価luminosityを推定することも避ける。

### m2 — 図の見出しとvalidationの範囲が実態より強い

図4はATLAS HL-LHC sensitivityの見出し、公式algorithm improvementの横線、誤差表示のない点から、公式再fitのように読める。小さなfooterだけで換算の限界を伝えるのは弱い（`../make_figures.py:154–193`）。図7のprocess総yieldの0.1–0.6% closureはPDFからの総量抽出を支えるが、bin内shape、origin fraction、d0 joint templateを検証しない（`../extract_latest.py:94–105`、`../hepdata_inputs.py:112–118`）。**対処：**図4の主題を「surrogate gainの公開値への換算」とし、scenarioと不確かさを可視化する。図7は「総yield closure」と限定する。図2は10 sub-bins、nominal fitは6である点もcaptionに記載する（`../make_figures.py:83`、`../study.py:77`）。

### m3 — fit収束の判定を数値の採用条件にしていない

Asimov μ=1のunconditional NLLを生成点で評価する手法自体は妥当。問題はμ=0のfitから`fun`だけで最小を選び、`success`がfalseでもZを返せる点である（`../projection.py:300–349`）。同時更新でL-BFGS-B二startにMinuit二startも加わったが、成功したfitだけを採用する条件はまだない。start間の一致は局所極値・bound hit・勾配の監査に代わらない。**対処：**全variantの成功flag、勾配、bound hitを示す。失敗fitを採用していると確認されたわけではなく、提供表には判定に必要な詳細がない。

### m4 — C1の「TauSpinによる増分」の比較対象を明示する

legacyではTauSpin対情報なしが+1.47%だが、textbook arm対情報なしは+0.92%、TauSpin対textbookは約+0.545%（`summary.md:46–47`）。後者が既存polarisation observableに対する追加効果である。これもATLAS分類器がtextbook armと同じ情報を使うという実証ではない。**対処：**「spin情報を追加する効果」と「TauSpinが従来observableを上回る効果」を分ける。

## 2404.12660のlepton d0/TTVA：一次資料の追跡結果

要求された `|d0/σ|<5 (e), <3 (μ), |z0 sinθ|<0.5 mm` が2404.12660のlep-hadで実際に使われたかは、**今回確認した公開一次資料だけでは確定できない**。標準値を知っていることと、特定解析が採用したことは別である。

| 追跡先 | 確認できたこと | 確定できないこと |
|---|---|---|
| [2404.12660 §4、Ref.37](https://arxiv.org/html/2404.12660v2) | 物体再構成・識別は2209.10910と同じ。§4.1にも独立d0 thresholdの記載なし。 | この継承だけから標準TTVA適用の有無は決まらない。 |
| [2209.10910 §4・§5.1/Table 2、Ref.133/134](https://arxiv.org/html/2209.10910) | electron likelihood ID、signal Tight electron/Medium muonを指定。参照先は1908.00005と2012.00578。 | 物体定義・Table 2に独立の3σ/5σ、z0 thresholdが明記されていない。§7のmuon track-to-vertex効率不確かさへの言及も、cut値・electronへの適用を確定しない。 |
| [1908.00005 Table 1・§6.1](https://arxiv.org/pdf/1908.00005) | electron likelihoodにd0とd0 significanceを使う。 | ID内部の利用は独立の5σ TTVA cutと同じではない。 |
| [2012.00578 §5.1.2・§5.2](https://arxiv.org/html/2012.00578) | Medium IDとvertex associationを別に定義。§5.2はmuonの3σとz0 0.5 mm、beam-widthを含むσを記載。 | Mediumを要求しただけで§5.2のvertex cutも自動的に要求されたとは結論できない。 |
| [ATLAS ttH 1712.08891 §4/Table 2](https://arxiv.org/pdf/1712.08891) | e 5σ・μ 3σの標準的なprompt選択の実例。 | 別解析の実例をHHの適用証拠にはできない。 |
| [最新ATLAS 2607.26879 §4](https://arxiv.org/html/2607.26879) | 横impact-parameter選択を課さない方針を明記。e/μのz0 0.5 mmを記載。 | 旧projectionの選択、またはID内部のIP変数除去を証明しない。 |

従ってpacketでは「旧ATLASも標準TTVAを使った」「旧ATLASはd0情報を全く使っていない」のどちらも断定しない。旧campaignの公開選択設定または解析担当者による設定確認が残る。少なくともelectron IDに含まれるIP情報の扱いは、gainの評価条件として必須である。

## 新規性：寿命識別は既知、HH→bbττでの専用追加利用は未確定

**広い意味の『τ→ℓとprompt ℓを横impact parameterで識別する』新規性は成立しない。** ATLASは2007.14040でprompt W→μとW→τ→μをmuonの|d0|・pT template fitで分離した。ただし絶対d0であり、packetと同じPV基準の有意度ではない。prompt shapeをZ→μμで校正する点も重要な先例である。[ATLAS R(τ/μ)：選択・d0校正・統計解析の各節、Fig.1、Methods](https://arxiv.org/html/2007.14040)。

| 調査対象・一次資料 | IPの用途／HHへの含意 |
|---|---|
| [ATLAS bbττ 2209.10910 §4–5/Table 3](https://arxiv.org/html/2209.10910)、[2404.12660 §4.3](https://arxiv.org/html/2404.12660v2)、[2607.26879 §4・§7/Table 2](https://arxiv.org/html/2607.26879) | light-lepton d0 shapeでdirect-W背景を除去する専用手法の明示は確認できない。ただしID内の情報利用とkinematicsとの相関はある。 |
| [CMS bbττ 2206.09401 §4/Table 1・§5](https://arxiv.org/html/2206.09401) | e/μにdxyの絶対値<0.045 cm、dzの絶対値<0.2 cm。DNNは26 featuresで、本文は主要入力の例示。完全な入力不使用の証明にはならない。 |
| [CMS最新HIG-25-008公式結果、2026-08-06](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/HIG-25-008/index.html)、[PAS全文 §5/Table 1・§6（pp.7–8）](https://inspirehep.net/files/cb6ce0aced049563b0ccca6510e0e01d) | Run 3 172 fb⁻¹のbbττ。e/μの同じdxy/dz cutを確認。DNN入力の説明はmissing-pT/covariance、four-momenta、tagging scores、rapidity gaps等で、light-lepton d0 shapeの専用利用は明示されない。tagging内部を含む情報の完全な未使用は断言しない。 |
| [CMS H→ττ 2204.12957 §4・§7.1](https://arxiv.org/html/2204.12957) | 同じlight-lepton dxy/dz cut。列挙された14 physics inputsにはIPがない。DeepTauのIP利用はτh IDであり、light-leptonのτ→ℓ識別と別。 |
| [CMS ttH 2011.03652 §4](https://arxiv.org/pdf/2011.03652)、[ATLAS ttH 1712.08891 §4/Table 2](https://arxiv.org/pdf/1712.08891) | CMS leptonMVAはdxy/dz等でheavy-flavour nonpromptを除去し、W/Z由来とleptonic τ由来を同じprompt classに含める。ATLASの3σ/5σも通常のnonprompt除去。τ→ℓ対direct-W専用識別とは区別する。 |
| [CMS R(J/ψ) 2408.00678：IP3D定義・signal extraction](https://arxiv.org/html/2408.00678) | Bc→J/ψτ→J/ψμとdirect μを区別するためthird-muon IP3D significanceを使う。基準はJ/ψ vertexで、PV-d0によるprompt-W除去の直接の先例ではない。 |
| [ATLAS H→ττ CP 2212.05833 §5/Tables 4–5](https://arxiv.org/pdf/2212.05833)、[CMS CP 2110.04836 §6.6.3・§7・§9](https://arxiv.org/html/2110.04836) | ATLASはe d0 significance>2.5、μ>2.0等をCP感度分類に使う。CMSはrefitted PVに対する3D IP significance>1.5等。CP角測定の用途であり、HHのprompt背景除去の実証ではない。 |
| [T. Lange博士論文、2022、p.91/Fig.6.16・Table 6.17付近](https://inspirehep.net/files/92f1a0e9851b9a26ee13555e7bc27603) | CMS HH multilepton **3ℓ+1τh**で、leptonic τ由来leptonの識別を目的にmaximum lepton IPを検討したが、BDT性能をさらに改善せず最終入力に採用しなかったと記載。bbττではなく、collaborationの確定結果でもない。『HHでの検討自体が初めて』には反例。 |
| [Hagiwara・Ma・Mori 1609.00943：h→ττ CP再構成、pp.1–3](https://arxiv.org/pdf/1609.00943)、[Desch・Was・Worek hep-ph/0302046 §4–5](https://arxiv.org/pdf/hep-ph/0302046) | τ decayのimpact-parameter方向をCP/polarimetryに利用する現象論。HH→bbττのdirect-W背景除去による感度増分を示す資料ではない。 |

CMSの450 µm cutは、τ→ℓの変位tailを原理的には切る。しかしτのcτから受容損失を直接計算できず、pT・boost・decay angle・基準vertexに依存する。10–20 µmの分解能を例にすると450 µmは約22–45σで、ATLASの3σ/5σと同じ制約ではない。cut単独のτ→ℓ効率低下を示す一次数値は確認できなかったため、「CMS標準選択が寿命情報を大幅に消している」とは言えない。

**TauPolarisとの差（1行）：** [2608.10961 §III・§IV.1/Table I・§VII/Fig.12](https://arxiv.org/html/2608.10961)はsingle-H/Z試料でIPをneutrino・polarimetric-vector再構成へ利用する研究で、本文中にHH感度の投影は確認できず、packetのlight-lepton寿命によるdirect-W除去とは評価対象が異なる。

検索した公開本文・結果ページでは、**HH→bbττのτℓτhにおいて、同じ既存選択・分類器の後に残るlight-lepton d0 significanceを専用に追加し、tt̄/W+jetsのdirect lepton除去から増分を定量化した先行実験解析・HL-LHC投影は確認できなかった**。これは不存在の証明ではない。内部解析、分類器実装内の全feature、別名のtrack/IP featureは未確認。許容される新規性の表現は「既知のτ寿命情報について、HH→bbττの既存選択後に残る追加効果を調べる」であり、「世界初」「既存HH解析は未使用」は保留する。

## ITk・pileup 200：公開値との比較

| 一次資料・位置 | 試料・幅の定義 | 確認できたd0 resolution |
|---|---|---|
| [Pixel TDR ATLAS-TDR-030/CERN-LHCC-2017-021、§3.1.2 p.44・Fig.3.6 p.45](https://inspirehep.net/files/a14abc21829f07dbf8e27329ea919d2d) | single μ、50×50 µm²、digital clustering。reco−truth residualのcore RMS。図にPU値の明記なし。 | 本文はpT=100 GeV、ηの絶対値<3で約10 µm。**Fig.3.6をPU200と断定しない。** Fig.3.7ではanalogue clustering・別pitchを区別。 |
| [IDTR-2023-01、2412.15090、JINST 20 (2025) P02018、§5.4・Fig.23b](https://arxiv.org/pdf/2412.15090)、[公式画像](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/IDTR-2023-01/fig_23b.png) | single μ、pT=100 GeV、**PU0**、layout 03-00-00。反復3σ除外によるcore SD。 | 図の概算：ηの絶対値<2.5で3.5–4.5 µm、3付近で6、3.5で9–11、3.8で15–20、4付近で28–30 µm。 |
| [2026年ATLAS公開EF性能図集](https://twiki.cern.ch/twiki/bin/view/AtlasPublic/EFTrackPerfTechChoicePlots)、[d0 vs truth pT](https://twiki.cern.ch/twiki/pub/AtlasPublic/EFTrackPerfTechChoicePlots/CPU_resolution_d0_vs_truth_pt.png)、[d0 vs truth η](https://twiki.cern.ch/twiki/pub/AtlasPublic/EFTrackPerfTechChoicePlots/CPU_resolution_d0_vs_truth_eta.png) | 14 TeV tt̄、**PU200**、layout 03-00-01、truth pT>1 GeV。黒C-000がdefault offline reference。EF候補のC-100/C-230等とは区別。 | 黒曲線の目視概算を下表に示す。lepton限定、幅推定法、η spectrum、PV基準の条件はcaptionから確定できない。 |

| pT [GeV] | PU200 tt̄ offline黒曲線 [µm、目視概算] | packetの式 [µm、計算] | electron×1.2 [µm、計算] |
|---:|---:|---:|---:|
| 10 | 約16（15–17） | 18.03 | 21.63 |
| 20 | 約11（10–12） | 12.50 | 15.00 |
| 50 | 約10（9–11） | 10.44 | 12.53 |
| 100 | 約9–10 | 10.11 | 12.13 |

式はPU200のinclusive track公開曲線と概ね同程度である。一方、PU0中央100 GeV muonのcore幅とは約2.5倍違い、forwardではη依存が大きい。試料・layout・幅定義が違うため、これをそのまま「packetは保守的／楽観的」と結論しない。**PU200でlepton限定のd0 significance tailを検証した式ではない**ことが比較からの確実な結論。

また `0.98 N(0,σ²)+0.02 N(0,(4σ)²)` というmixtureを意味するなら、全RMSは1.140σ、|d0|>3σ/4σ/5σのprompt確率は計算上1.171%/0.641%/0.423%となる。「2%が4σを超える」モデルとは異なる。これは説明文の解釈に基づく計算で、外部prepared dataのtail生成実装を検証した値ではない。図3のnominal prompt >5σ tailが約0.4%であることとは整合するが、ITkで実証されたtailではない。

## 主要図の画像確認

| 図 | 画像から確認したこと | 読み取りの限界・必要な修正 |
|---|---|---|
| `fig1_composition.png` | signal-like binにもZ+HF、single-H、fakeが存在し、組成によってspin増分が小さくなる説明は定性的に可能。 | τhのtrue/fake分類と、light-leptonのτ/direct-W/nonprompt起源を混同しない。 |
| `fig2_spin_templates.png` | 再構成spinのH/background shape差は限定的で、exact armとの差が見える。 | 真のτから作ったfake proxy。10-bin表示と6-bin nominal fitを区別する。 |
| `fig3_lifetime.png` | τ→ℓの>5σ tailは約0.23、nominal promptは約0.004、degraded promptは約0.011、synthetic nonpromptは約0.37。 | 大きな識別力は形状仮定に依存。高score top/fake起源点が欠け、誤差表示もない。 |
| `fig4_main_result.png` | 4.42σ、4.57–4.58σ等の換算値と公式改善scenarioを並べている。 | 公式再fitと誤認し得る見出し。換算であることを主表示にする。 |
| `fig5_robustness.png` | 寿命・joint増分がresolution、TTVA、origin mixtureで大きく変わる。 | 選択scenarioの範囲であり、信頼区間・下限ではない。 |
| `fig6_where_gain.png` | high-score側が寄与を支え、SLTのΣZ²は3.66→5.11、had-hadは12.03→12.50。 | known-background・no-profilingの説明図で、profile済みZ²の加法分解ではない（`../make_figures.py:227–250`）。LTTは未表示。 |
| `fig7_validation.png` | region総yieldの良好なclosureと、lep-hadのsystematic degradation mismatchが見える。 | 総yield closureからjoint shapeの正確さは導けない。 |

## 主張C1〜C3の最終判定と受け入れられる文言

| 主張 | 判定 | 理由 |
|---|---|---|
| C1：spinのみで公式結合4.26→4.29–4.30σ | **文言修正で可** | 小さい正の増分は提供toy結果として整合する。公式感度の実証、背景組成による因果説明、TauSpin固有の増分へ一般化しない。 |
| C2：jointで4.42σ、legacy/latestでstable、degradationでも4.31–4.38σ | **不可（現状の実験感度・頑健性の主張として）** | B1–B3が未検証。origin/tailの依存性が強く、scenario範囲に統計的保証がない。数値は仮定に条件付けたillustrationとして記述できる。 |
| C3：exact h＋寿命4.57σが情報上限 | **不可（上限の主張として）** | 完全なjoint情報・responseを扱っておらず、finite-bin oracleの上限性も未証明。benchmarkとして記述できる。 |

**C1の受け入れられる文言案：**

> 公開yieldとHH試料由来の応答を用いたsurrogate modelでは、再構成spin情報を追加したbbττ significance比はlegacy/latestで約1.015/1.012だった。公式HL-LHC値への経験的換算ではHH結合4.29–4.30σに相当する。本モデルでは高感度binの背景組成と弱いspin分離により効果は小さいが、これはATLAS workspaceで確認した感度向上ではない。

**C2に代えて受け入れられる文言案：**

> lepton寿命の識別情報は既知である。既存ID後の残余情報とscore相関が未検証の現在のtoyでは、寿命単独のbbττ比は1.041–1.048、spinと合わせると1.061–1.063となり、経験的な結合換算は約4.42σだった。選んだstressでは約4.31–4.38σへ変化する。この範囲は信頼区間や下限ではなく、ATLASでの実際の追加効果を確定するものではない。

**C3に代えて受け入れられる文言案：**

> 固定したspin-density、背景応答、寿命分解能、binningおよび換算方法の下で、exact-h armと寿命情報のoracle benchmarkは結合換算4.57–4.58σとなった。これは、このモデルの比較基準であり、完全なτ情報またはATLAS解析の感度上限ではない。
