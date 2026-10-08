# τ spinをHH→bbττへ導入するための公開文献・尤度資料調査

確認日：2026-10-08。数値の「期待」はSM信号を注入したdiscovery significanceと、background-onlyでのexpected upper limitを区別する。ダウンロードのHTTP status、Content-Type、URL、エラーは[data/fetch_manifest.json](data/fetch_manifest.json)に保存した。HTTP 200でもPDFではなくbot確認HTMLだった例がある。

## 1. 要約

1. ATLASの「4.3σ」の正本は[ATL-PHYS-PUB-2025-006](https://inspirehep.net/files/5cabc0e66dede649b8f6e37147e87a82)、Table 5の**4.26σ**（14 TeV、3000 fb⁻¹、baseline=S2）。No syst. unc.は5.98σ。
2. 同文書Table 7のκλ 68% CIは**[0.58, 1.48]**。Table 9でb-taggingとτhad IDの効率を各5%改善すると4.52σ、[0.61, 1.44]。
3. bbττの投影元は**arXiv:2404.12660**。2209.10910や2026年のTransformer解析ではない。2404.12660はbbττ単独解析で、ATLAS HH組合せは2406.09971。
4. [ATL-PHYS-PUB-2024-016](https://inspirehep.net/files/735311e4e52d37c9117fc7cde5b69aff)、Table 4：3 ab⁻¹のbaselineでτhadτhad／τlepτhad／合計=3.1／1.8／3.5σ、No syst. unc.で4.0／2.3／4.6σ。
5. [2504.00672](https://arxiv.org/pdf/2504.00672)、Table 2：CMS単独S2／S3=4.2／4.5σ、ATLAS+CMS=7.2／7.6σ。共同投影は一部チャンネルを両実験へ複製する仮定を含む。
6. 最新ATLAS[2607.26879](https://arxiv.org/html/2607.26879)は196 fb⁻¹、期待1.24σ（Auxiliary Figure 20）。12 SRの総収量表を取得したが、bin別数値データと公開尤度は確認できず、HEPData ins3184980は404。
7. legacyのHEPData **ins2155171、record 130794、Figure 8a–8c**から全44 binsのprocess別収量を取得。ただしLTTのTotal Background列とsignal規格化に不整合があり、公開pyhf JSONもない。
8. 最近接は**TauPolaris 2608.10961**と**BONN-IB-2014-03**。前者はh再構成、後者はreco H_ZをH/Z BDTに使用。「再構成hをHH分類器へ追加しHL-LHCのprofile likelihood感度を評価」の一致研究は本調査では見つからない。

## 2. A〜Dの確定事実と未確定事項

### A. HL-LHC HH投影の正本

#### A1. ATLASの正本とチャンネル別数値【確定】

[ATL-PHYS-PUB-2025-006、2025-03-04](https://inspirehep.net/files/5cabc0e66dede649b8f6e37147e87a82)はRun 2の6解析、126–140 fb⁻¹を組み合わせた投影。以下は**Table 5**の期待discovery significanceで、単位はσ。bbbbはresolvedとboostedを合わせた列であり、SM discoveryへのboosted単独寄与を表から分離できない。

|L′ [fb⁻¹]|系統シナリオ|bbγγ|bbττ|bbbb|Multilepton|bbℓ⁺ℓ⁻+E_T^miss|Combination|
|---:|---|---:|---:|---:|---:|---:|---:|
|2000|Run 2 syst. unc.（S1）|1.76|1.84|0.62|0.69|0.33|2.49|
|2000|Theory unc. halved|2.04|2.03|0.62|0.74|0.44|2.97|
|2000|Baseline（S2）|2.06|3.00|0.89|0.84|0.45|3.71|
|2000|No syst. unc.|2.23|3.76|1.44|0.98|1.31|4.88|
|3000|Run 2 syst. unc.（S1）|2.00|1.93|0.65|0.79|0.37|2.73|
|3000|Theory unc. halved|2.39|2.17|0.65|0.85|0.48|3.32|
|3000|Baseline（S2）|2.43|3.54|0.99|0.99|0.48|4.26|
|3000|No syst. unc.|2.73|4.60|1.76|1.20|1.60|5.98|

同じ条件のκλ **68% CI**は**Table 7**。取得した[単独表PDF](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PUBNOTES/ATL-PHYS-PUB-2025-006/tab_07.pdf)も開いて確認した。

|3000 fb⁻¹のシナリオ|bbγγ|bbττ|bbbb|Multilepton|bbℓ⁺ℓ⁻+E_T^miss|Combination|
|---|---|---|---|---|---|---|
|Run 2 syst. unc.|[0.35, 1.95]|[0.05, 2.34] ∪ [5.24, 6.21]|[−1.19, 6.81]|[−0.27, 4.79]|[−3.02, 9.89]|[0.39, 1.71]|
|Theory unc. halved|[0.42, 1.79]|[0.28, 2.13] ∪ [5.42, 6.10]|[−1.13, 6.80]|[−0.16, 4.76]|[−2.10, 9.17]|[0.52, 1.59]|
|Baseline|[0.40, 1.79]|[0.42, 1.68]|[−0.53, 6.06]|[−0.05, 4.65]|[−2.10, 9.17]|[0.58, 1.48]|
|No syst. unc.|[0.47, 1.70]|[0.59, 1.48]|[0.12, 2.57]|[0.10, 4.61]|[−0.06, 2.95]|[0.71, 1.33]|

これらを95% CIとして引用してはいけない。bbττ単独投影[ATL-PHYS-PUB-2024-016](https://inspirehep.net/files/735311e4e52d37c9117fc7cde5b69aff)、**Table 5**の3000 fb⁻¹におけるκλ **95% CI**はbaselineで[−0.1, 2.7] ∪ [4.5, 6.4]、No syst. unc.で[0.3, 2.1]。同noteのabstractはstat onlyを[0.2, 2.1]と記しており、本文Figure 8とTable 5の[0.3, 2.1]と一致しない。ここでは表の値を採った。

τ decay channel別は同noteの**Table 4**に確定値がある。τlepτhadはSLT/LTTを合算した値。

|L′ [fb⁻¹]|シナリオ|τlepτhad [σ]|τhadτhad [σ]|Combined [σ]|
|---:|---|---:|---:|---:|
|2000|No syst. unc.|1.9|3.2|3.8|
|2000|Baseline|1.5|2.6|3.0|
|2000|Baseline with MC luminosity scaled|1.4|2.5|2.9|
|2000|MC luminosity scaled|1.3|2.1|2.4|
|2000|Theoretical unc. halved|0.9|1.8|2.0|
|2000|Run 2 syst. unc.|0.9|1.7|1.8|
|3000|No syst. unc.|2.3|4.0|4.6|
|3000|Baseline|1.8|3.1|3.5|
|3000|Baseline with MC luminosity scaled|1.7|3.0|3.4|
|3000|MC luminosity scaled|1.6|2.4|2.7|
|3000|Theoretical unc. halved|1.0|1.9|2.2|
|3000|Run 2 syst. unc.|0.9|1.8|1.9|

#### A2. 元解析と外挿規則【確定】

2025-006の**§2、references [36], [43], [51]**によって出典の連鎖が確定する。

|用途|文献|この調査での位置付け|
|---|---|---|
|Run-2旧bbττ|[2209.10910](https://arxiv.org/pdf/2209.10910)|139 fb⁻¹、3カテゴリ、最終binのTable 5とHEPDataあり|
|投影元のRun-2更新bbττ|[2404.12660](https://arxiv.org/pdf/2404.12660)|140 fb⁻¹、ggF Hi/LoとVBFを導入、BDT。HH組合せではない|
|ATLAS Run-2 HH組合せ|[2406.09971](https://arxiv.org/abs/2406.09971)|2025-006の組合せ手順の基礎|
|bbττの単独投影|[ATL-PHYS-PUB-2024-016](https://inspirehep.net/files/735311e4e52d37c9117fc7cde5b69aff)|上の2404.12660を外挿|
|ATLAS HH組合せ投影|[ATL-PHYS-PUB-2025-006](https://inspirehep.net/files/5cabc0e66dede649b8f6e37147e87a82)|4.26σの正本|
|2026年の最新bbττ|[2607.26879](https://arxiv.org/pdf/2607.26879)|Transformer、Run 2+3。2025投影の入力ではない|

2025-006 **§3、Table 1**：

- 全processの収量にL′/Lを掛ける。13→14 TeVのcross-section係数はggF HH 1.18、VBF HH 1.19、ggF H 1.13、VBF H 1.13、WH 1.10、ZH 1.12、ttH 1.21、Others 1.18。
- 最終discriminantとbinningは変更しない。ITkなどのPhase-2 upgradesによって、pile-up下でもRun 2相当以上のobject performanceを保つという仮定であり、ITkを通したHH full simulationの再解析ではない。
- bbττはRun-2 post-fit補正を一律に掛けず、**Z+HFに1.3**を掛ける（§3、bbττ段落）。baselineではこの正規化補正の不確かさを無視する。

同文書**Table 2**のbbττ列におけるuncertainty scale factor：

|uncertainty|baselineの係数|
|---|---:|
|Luminosity、electron/muon efficiency、JES/JER、E_T^miss|1.0|
|b-/c-jet b-tagging efficiency uncertainty|0.5|
|light-flavour b-tagging efficiency uncertainty|1.0|
|τhad efficiency（statistical）|0.0|
|τhad efficiency（systematic）、τhad energy scale|1.0|
|Fake-τhad estimation（statistical）|0.0|
|Fake-τhad estimation（systematic）|0.5|
|Signal theory、Others theory|0.5|
|κλ reweighting uncertainty、MC statistics|0.0|

**資料内の相違**：2025-006のTable 2はfake systematic=0.5だが、§3のbbττ段落はsystematic componentをRun 2と同じに保つと書く。2024-016 Table 2も0.5。workspaceが公開取得できないため、実装がどちらに従ったかまでは確定できない。またτ energy scaleのstatistical componentは本文で0とし、表にはenergy scale全体の1.0を掲げている。表の係数をそのまま、全NPへの一括操作と解釈するのも避ける。

2024-016 **§3**のMC luminosity scaledはMC template statisticsとdata-driven fake statisticsを√(L/L′)で縮める規則。2025-006 **§3**によればこれをbaselineに残す場合、bbττの期待有意度は**3.54→3.41σ**。MC統計を無視するかどうかは、h追加の効果を比べるときにも合わせる必要がある。

object efficiencyの改善は別の仮定。2025-006 **§5、Table 9**でb-tagging効率を相対5%改善すると組合せ4.44σ、τhad IDのみ5%で4.34σ、両方で**4.52σ**。baselineの4.26σに既にこれらの改善が含まれているわけではない。同表の両方改善・No syst. unc.は6.50σ。

#### A3. CMSとATLAS+CMS【確定】

[2504.00672 / ATL-PHYS-PUB-2025-018、§3、Table 2](https://arxiv.org/pdf/2504.00672)はEuropean Strategy向け共同総括。文書中のκ3は本調査でいうκλに対応する。以下は同表の原数値で、各実験あたりのルミノシティ。

|チャンネル／組合せ|2 ab⁻¹ S2 ATLAS / CMS|3 ab⁻¹ S2 ATLAS / CMS|3 ab⁻¹ S3 ATLAS / CMS|
|---|---|---|---|
|bbττ significance [σ]|3.0 / 1.9|3.5 / 2.4|3.8 / 2.7|
|bbγγ significance [σ]|2.1 / 2.0|2.4 / 2.4|2.6 / 2.6|
|bbbb resolved significance [σ]|0.9 / 1.0|1.0 / 1.2|1.0 / 1.3|
|bbbb boosted significance [σ]|— / 1.8|— / 2.2|— / 2.2|
|Multilepton significance [σ]|0.8 / —|1.0 / —|1.0 / —|
|bbℓ⁺ℓ⁻ significance [σ]|0.4 / —|0.5 / —|0.5 / —|
|各実験Combination significance [σ]|3.7 / 3.5|4.3 / 4.2|4.5 / 4.5|
|ATLAS+CMS significance [σ]|6.0|7.2|7.6|
|bbττ κ3 68% CI|[0.3,1.8] / [0.1,3.0]|[0.4,1.7] / [0.2,2.2]|[0.5,1.6] / [0.3,2.0]|
|各実験Combination κ3 68% CI|[0.6,1.5] / [0.4,1.7]|[0.6,1.5] / [0.5,1.6]|[0.6,1.4] / [0.6,1.5]|
|ATLAS+CMS κ3 uncertainty|−32% / +37%|−27% / +31%|−26% / +29%|

**7.2σは独立のATLAS 4.3σとCMS 4.2σを単純に組み合わせた数字ではない。** 同文書§3とTable 2の†印では、ATLASのbbττ、Multilepton、bbℓ⁺ℓ⁻とCMSのbbbb resolved/boostedを両実験へ採用し、該当チャンネルを6 ab⁻¹へ外挿する。bbγγは各実験の独立投影。CMS Run-2 bbττのtrigger制限はRun 3 trigger改善でATLAS相当になるという仮定を説明している。S3はb-taggingとτhad reconstruction efficiencyを各5%改善したシナリオ。

CMS単独正本は[CMS-NOTE-2025-006、2025-03-17、v2 2025-03-26](https://cds.cern.ch/record/2928096)。CDSのabstractは主要シナリオで4.5σ、κλの68%精度50%と記す。この4.5σは共同総括のS3列と一致する。

#### Aで確定できなかったことと試行

- **CMS-NOTE本文のS1/stat only、細かい外挿NP規則**：[CDS PDF](https://cds.cern.ch/record/2928096/files/NOTE2025_006.pdf)とrecordを開いたが、HTTP 200の“Making sure you're not a bot!” HTML。確認画面の回避は行っていない。CMS public-resultsのNOTE-2025-006配下でNOTE-2025-006.pdf、CMS-NOTE-2025-006.pdf、preliminary-results側NOTE2025_006.pdfを試したが404。INSPIREのreport number／title検索も該当recordを返さず。したがってCMSのS2/S3数値は**開いて取得できた共同総括Table 2**で確定し、本文未取得のnoteの表番号は捏造しない。
- **ATLAS最新解析2607.26879を元にしたHL-LHC投影**：最新paper、公開ページ、2025/2026投影検索を確認した範囲では見つからない。公開Figure Aux 31/32は現ルミノシティ付近での改善の比較であり、Phase-2 systematic scenarioを与えたHL-LHC投影ではない。
- **HH組合せκλの95% CIを全シナリオで表から得ること**：2025-006はTable 7に68% CIを数値で提供し、Figure 4に95%閾値を示す。95% CIを68% CIの定数倍から作ることはしなかった。

### B. 最新HH→bbττ実解析

#### B1. ATLAS 2607.26879の選択とMVA【確定】

一次資料は[paper HTML](https://arxiv.org/html/2607.26879)、[PDF](https://arxiv.org/pdf/2607.26879)、[HIGP-2024-37公開ページ](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/)。

- **§3**：Run 2 140 fb⁻¹（13 TeV）＋Run 3 56 fb⁻¹（13.6 TeV、2022–2023）。§4のb-taggingはGN2、τhad IDはLoose RNNで1-track／3-track効率85%／75%。τhadはp_T>20 GeV、|η|<2.5、1.37<|η|<1.52を除外、1または3 tracks、|charge|=1。
- **§4**：electron/muonに|z₀ sinθ|<0.5 mm、**独立のtransverse d₀ cutを適用しない**。τ由来leptonの受容率を保つ目的が本文に明記されている。PV関連選択が存在することと、τのIP vectorが分類器入力であることは別。
- **§5.1**：τhadτhadはexactly two opposite-charge τhad、electron/muon veto、m_ττ^vis>40 GeV、m_ττ^MMC>60 GeV、40<m_bb<210 GeV。STT、DTT、Run-3 DBTをtrigger bucketで排他的に扱う。DTTの代表offline閾値は40/30 GeV、追加jet p_T>50 GeV。2023 delayed streamには低いτ thresholdのDTTなどを追加。
- **§5.2**：τlepτhadはexactly one e/μ＋one opposite-charge τhad、m_ττ^vis>40 GeV、m_ττ^MMC>60 GeV、40<m_bb<150 GeV。**SLTのみを使用し、以前のLTTは感度改善が小さく使用しない。** electron offline p_T>27 GeV（2015は25 GeV）、muonはtrigger閾値20–26 GeVより1 GeV以上高くする。
- **§7、Figure 4**：τhadτhad／τlepτhadごとにTransformerを学習。S_cat=0.1でVBFを分け、残りのm_HH>350 GeVをSR Hi、その他をSR Lo。τhadτhadのSR Hiはmain streamのみ。Run 2/3を分け、**2 channels×3 regions×2 runs=12 SR**。旧解析のτlepτhad SLT/LTT区分を最新解析へ移植してはいけない。
- **§9**：Z+HF CRとtW CRをrunごとに追加し、4 CR。SRではscore、Z+HF CRではm_ℓℓ、tW CRではm_bℓをfit。score bin数は9–15、各binのbackground statistical uncertainty<20%、expected background≥3という条件で最適化する。

**Table 2のMVA入力を全項目列挙**：

|入力群|入力|
|---|---|
|共通global|m_HH（MMC）、m_bb、ΔR_bb、m_jj、ΔR_jj、η_VBFj0·η_VBFj1|
|τhadτhad追加global|ΔR_ττ|
|τlepτhad追加global|ΔR(τhad,ℓ)、p_T,HH、η_HH、E_T^miss、ΔR(b₀,τhad)、ΔR(b₁,τhad)、m_bℓ、m_ττ^vis、p_T,ττ^vis、T₁（Topness、σ_t=σ_W=5 GeV）、H_T、Δη(bb,ττ)、Δφ(ττ,E_T^miss)、ΔR(b₁,ℓ)、m_T(τhad,p⃗_T^miss)|
|各object|η、φ、m、p_T、PCBT（GN2 pseudo-continuous b-tag score、100%–90%–85%–77%–70%–65%）、isTau、isJet、isBjet、isMET、isMMC、isLepton、validity flag|
|objectの種類|jets、τhad candidates、MET、MMC di-τ four-momentum、τlepτhadのlepton|

**この一覧にh、φ*_CP、τ decay-plane、τ constituent momenta、τ IP vector、τ SV vectorはない。** τhad objectのmassも0に設定する。GN2やτ ID内部のtrack情報が間接的に作用する可能性はあるが、このevent MVAが再構成τ spinを明示的に使用しているという証拠にはならない。

#### B2. 収量・尤度と、どこまで分解できるか【確定】

最新ATLASの**Figure 7a–f / 8a–f**は12 SRのpost-fit score分布。**Auxiliary Table 01/02**は各SRを全score binで積分したprocess収量で、数値は§3の表へ転載した。Auxiliary Table 03はZH/ZZ系統分解、04はVBF signal-strength系統分解。**score binごとの数値表ではない。** 総収量をfit最終binの背景組成として扱えない。

最新の公開process区分：

- τhadτhad：tt̄（true τhad）、jet→τhad fake（tt̄、少量tW/W+jetsを含む）、lepton→τhad fake、multijet、tW、Z+HF、Single Higgs boson、Other。
- τlepτhad：true-τhad tt̄、data-driven fake-τhad（tt̄主体、tW/W+jets/multijetも含む）、lepton fake、tW、Z+HF、Single Higgs boson、Other。
- **single HのggF/VBF/ZH/ttH別bin収量は公開表で分かれていない。** Figure Aux 26–30はZH/ZZとclassifierの相関を示す2D分布で、全processのnominal histogramやNP variationの代用にはならない。

投影元[2404.12660の公開ページ、HDBS-2019-27](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HDBS-2019-27/index.php)は、**HEPData ins2779337（record 151276）**を直接リンクする。32表はlimit、κλ/κ2Vの1D scan、2D contours、EFT constraints、acceptance×efficiencyであり、Figure 6の9カテゴリのprocess別score histogramを含まない。全submission archiveとresourcesを確認し、HistFactory/pyhf modelも見つからなかった。κλのexpected scanは**“Combined $−2\ln\Lambda$ vs $\kappa_{\lambda}$ exp”、table id 1714476、Figure 8a**で取得できるが、これは既存fitの結果であって新変数を加えられるworkspaceではない。

同公開ページの**Auxiliary Table 01/02**は、ggF Lo/Hiの6 BDT／VBFの3 BDTの入力変数一覧。**Auxiliary Table 03**はSM/EFT benchmarkのacceptance×efficiency。ここにもbin別yieldはない。本文**§6**は9カテゴリ＋m_ℓℓ CR、floating tt̄/Z+HF NF=**0.96±0.03／1.34±0.08**、期待discovery significance **0.75σ**を報告する。Figure 6bはτhadτhad high-m_HHのpost-fit分布（HHはμ=2.2）で、直接PDFを保存した。

legacy [2209.10910](https://arxiv.org/pdf/2209.10910)は**Table 5**だけでなく[HEPData ins2155171](https://www.hepdata.net/record/ins2155171?format=json)に**Figure 8a（14 bins）、8b（15 bins）、8c（15 bins）**の数値がある。record番号130794、version 1、DOI 10.17182/hepdata.130794.v1。取得したsubmission archiveには53 entries、submission.yamlのadditional_resourcesは補助資料ページだけで、pyhf/workspace/likelihood JSONは含まれない。

**取得データの不整合を検査した結果**：

1. Figure 8a/bのbackground componentの和はTotal Backgroundに丸め誤差約0.002 events以内で一致する。
2. Figure 8cは一致しない。最終binはTop=2.371、fake=0.795、Z+HF=1.549、Other=0.333、SM Higgs=0.402、和=**5.450**なのにTotal Background=**2.371**。これは丸めでは説明できない。YAML直ダウンロードとsubmission archiveのraw YAMLでも同じ値。Table 5本文ではB=6±1、Z+HF=1.7±0.4。
3. Figure 8aのHH列は“SM HH at expected limit”、最終bin5.7706809。一方、最終paperのFigure 8 captionはcombined expected limit μ=3.9、Table 5のSM signalはggF 1.58＋VBF 0.0227。5.7706809/3.9=**1.4797**で、Table 5の**1.6027**と一致しない。8b/cにも差がある。旧version由来かどうかは推測にとどめ、**44-bin表ではraw signalを保持してSM signalに換算しなかった**。
4. HEPData Table 5 descriptionは“last two bins”と書くが、最終paperのTable 5 captionは“most signal-like bin”。掲載されたbackground数値は本文Table 5と対応する。説明文の不整合を黙って解消しない。

このため、legacyはbackground compositionの具体化には役立つが、取得したnominal yieldだけで実験のprofile likelihoodを再現したとはいえない。単独processのsymerror=0が多数あるのも、誤差がゼロという物理主張ではなく、post-fit total uncertaintyを別列に持つ表形式である。

#### B3. 期待感度とS/B【確定／限定】

最新ATLAS **[Auxiliary Figure 20](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/figaux_20.pdf)**の数値はRun 2／Run 3／Combinedの**期待1.11／0.60／1.24σ**（観測2.25／1.74／2.64σ）。abstractの期待1.2σはこの丸め。**[Auxiliary Figure 31](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/figaux_31.pdf)**はpre-fit、unconditional post-fit、μ_HH=1 conditional post-fitのAsimov定義を比較する。最新を元に外挿する場合には、この定義の差も揃える必要がある。

最新の個別τ channel・Hi/Lo/VBFの期待discovery significanceの数値表は見つからなかった。**Table 4**の個別値はbackground-onlyでのexpected 95% upper limitであり、σではない：

|dataset／channel|expected μ_HH upper limit（95% CL）|
|---|---:|
|τlepτhad Run 3|7.9|
|τlepτhad Run 2|4.6|
|τhadτhad Run 3|5.5|
|τhadτhad Run 2|3.0|
|Run 3 combined|4.3|
|Run 2 combined|2.6|
|Run 2+3 combined|2.4 (+1.2/−0.7)|

最感度binの厳密なSM S/Bは最新では未取得。Figure 7cのRun-2 τhadτhad SR Hi最終binを実際に開くと、single HとZ+HFが大きいことは読めるが、図の積み上げHHは**μ_HH=2.59**、別線はggF/VBF HH×20。Figure 10は元のfit binsをlog₁₀(S/B)で再集約した図で、元のカテゴリ×binのS/B表ではない。画像からの目測値を精密なfit入力として転載しなかった。

legacyの最終binはTable 5から確定できる。SはggF+VBFのSM yield、Bはpost-fit背景で、以下のS/Bはその数値からの算術計算：

|カテゴリ|S_SM|B|S/B（算出）|single H / B（算出）|Z+HF / B（算出）|
|---|---:|---:|---:|---:|---:|
|τhadτhad|1.6027|6.1|0.263|27.9%|37.7%|
|τlepτhad SLT|0.7775|6|0.130|18.3%|25.8%|
|τlepτhad LTT|0.25455|6|0.0424|6.7%|28.3%|

これは2209.10910の値であり、2404.12660を元にしたHL-LHC projectionや2607.26879の最感度bin組成ではない。

#### B4. 系統誤差とfree normalization【確定】

最新ATLAS **[2607.26879、§9、Table 3](https://arxiv.org/html/2607.26879)**：Z+HF、true-τ tt̄、tWのnormalizationsはunconstrainedで、Run 2/3に独立。MC background template statisticsには“light” Beeston–Barlowを使用。**Table 3**のpost-fit NF：

|run|Z+HF|tt̄|tW|
|---|---:|---:|---:|
|Run 2|1.11±0.08|0.97±0.04|1.34±0.16|
|Run 3|1.21±0.11|0.96±0.05|1.05±0.19|

**Table 5**はμ_HH=2.6でのσ_μHHのgroup breakdownであり、個々のNPのpull/ranking plotとは異なる：

|source|σ_μHH|
|---|---:|
|Total / Statistical / Systematic|1.20 / 0.91 / 0.76|
|Experimental / Modelling|0.20 / 0.71|
|HH / Other SM Higgs|0.54 / 0.25|
|Background template statistical / tW|0.21 / 0.21|
|Top / Z+jets|0.12 / 0.07|
|Jet / τhad / Flavour tagging|0.14 / 0.13 / 0.04|

**§10**は主要順をggF HH cross-section（QCD scalesとtop-quark mass scheme）、ggH+HF、tW–tt̄ interference、最感度binのtemplate statisticsと説明する。statisticalがtotal varianceの60%、signal/background modellingが35%、experimentalが3%。系統を除くとexpected upper limitが21%改善するが、この比較でもfloating normalizationとbackground template statisticsは残す。

legacy **[2209.10910、§7–8、Table 4](https://arxiv.org/pdf/2209.10910)**もtt̄とZ+HF normalizationをfitで決める。non-resonantのrelative uncertainty contributionはstatistical 81%、systematic 58%、single H modelling 29%、Top 24%、Z→ττ+HF 9%、fake τ 8%。これらは同論文の定義による標準偏差の相対寄与であり、足して100%になる割合ではない。

#### B5. CMSの最新bbττとRun 2【確定】

最新公開結果は[CMS-PAS-HIG-25-008、2026-08-06公開](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/HIG-25-008/)。公開ページのSummary、**Table 1–3、Figure 4**を取得した。Run 3は2022–2024の172 fb⁻¹、Run 2 138 fb⁻¹と合わせ310 fb⁻¹。

- Summary/Table 3：Run 3 observed(expected) μ_HH upper limit=6.6(3.9)、VBF=81(85)。Run 2+3=**4.0(3.0)**、VBF=62(70)。Run-2を同枠組みで処理した値は3.5(5.3)。
- **公開Summary**：Run 2+3 κλ 95% observed(expected) interval=**(−2.5,9.4)/(−1.5,8.4)**、κ2V=**(0.02,2.1)/(−0.1,2.2)**。[Figure 6](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/HIG-25-008/CMS-PAS-HIG-25-008_Figure_006.pdf)はRun-3のcross-section limit curveを示すが、境界数値は図にannotateされない。Run-3だけの厳密な区間は本調査では未確定。
- Table 1：τeτh、τμτh、τhτhとboosted τhτh。e/μは|d_xy|<0.045 cm、|d_z|<0.2 cm、τhは|d_z|<0.2 cm。resolved ττはΔR>0.5、boosted τhτhはΔR<0.8。DeepTauのvsJet Medium、vsEle VVLoose、vsMuon VLoose/Tightと、Boosted DeepTauを使用。
- Figure 4 caption：最終fitはDNN score。resolved 2b、resolved 1b1jを各4 τ final statesに分け、boosted bbとVBFはmerged Xτh channel。図のsignalはRun-3 best-fit μ=2.6へ規格化。
- [**Figure 5**](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/HIG-25-008/CMS-PAS-HIG-25-008_Figure_005.pdf)のRun-3カテゴリ別**expected μ_HH upper limit**はresolved 2b **4.2**、resolved 1b1j **15**、bb-boosted **21**、VBF **23**、combined **3.9**。τ final statesをまとめた値で、expected discovery σではない。
- Table 2：relative contribution to Δμ_HHはstatistical overall 74%、bin-by-bin variations 35%、HH theory 31%、single-H theory 27%、fake factors 25%、background theory 14%、b-tagging 10%、lepton corrections/triggers 9%、jet corrections 7%。groupの重なり／相関があるため単純な百分率和ではない。

[CMS Run-2 2206.09401 / HIG-20-010](https://arxiv.org/pdf/2206.09401)、**§4–5**：τeτh、τμτh、τhτh、resolved 1b/2b、boostedとVBF categories、最終DNN。26 featuresを100以上の候補から選び、主要入力としてyear、ττ decay mode、VBF jet数、DeepJet score、m_HH/m_ττ/m_bb、visible momenta、η–φ distancesを列挙する。**26変数の完全な一覧は本文に掲載されていない。** explicit hやφ_CPの使用は確認できない。§4のDeepTauは高水準τ変数とPF constituentsを使用する。Run-2論文のobserved(expected) μ_HH upper limitは3.3(5.2)、κλ 95%は[−1.7,8.7]（[−2.9,9.8]）。

#### Bで確定できなかったことと試行

- 最新ATLAS：[HEPData ins3184980?format=json](https://www.hepdata.net/record/ins3184980?format=json)を指定のAccept: application/jsonでcurl取得→**404**。INSPIRE metadataにはHEPDATA identifierがなく、公開ページの全リンクにもHEPData／pyhf／HistFactory JSON／ROOT templateのリンクは見つからない。公開Auxiliary Table 01–04は全て直接取得・内容確認済み。bin別yieldやsingle-H production-mode別yieldの表ではなかった。
- 投影元2404.12660：公開ページのINSPIRE linkにある[ins2816777?format=json](https://www.hepdata.net/record/ins2816777?format=json)は**404**だったが、同じ公開ページのData pointsは別番号の[**ins2779337?format=json**](https://www.hepdata.net/record/ins2779337?format=json)を指し、**取得成功**。record **151276、version 1、32 tables**。full YAML archiveは34 entries（directory、submission.yaml、32 data YAML）で、additional_resourcesは公開補助資料ページのみ。limit、coupling scan／contour、acceptance×efficiencyを収録するが、Figure 6のscore-bin yieldやpyhf／HistFactory workspaceは含まれない。**profile-likelihood scanの数値公開と、新しいdiscriminantを追加できるlikelihood modelの公開は異なる。**
- legacy：record JSONと全submission YAML archiveを取得し、resourcesとfile listを確認。**公開された当該recordにpyhf JSONはない**。これをATLAS全体で尤度公開がないという一般論には広げない。
- HEPDataのarXiv番号検索はHTTP 200でも、最新番号に対して無関係な古いrecordsを返した。検索のtotal=20を「20件の該当解析」とは扱っていない。
- CMS最新：[HIG-25-008-pas.pdf](https://cds.cern.ch/record/2968133/files/HIG-25-008-pas.pdf)→bot確認HTML（HTTP 200）。CMS public-results側の同名PDFは404。ただしTable 1–3とFigure PDFは取得できた。HEPDataリンクは公開ページ上でins_HEPDATA_INS_という未置換placeholder。検索で当該resultを確定できず、**全MVA入力一覧、bin別process yield、free background normalization、カテゴリ別期待σは未確認**。PAS本文を読んだかのように推定していない。

### C. τ spin・polarisationの先行研究

#### C1. HHへの利用、tt̄/Z/Hのspin【確定と適用範囲】

確認した最新ATLAS、投影元ATLAS、CMS Run-2のHH解析は、kinematics、mass、tagging scoreを使う。再構成h／明示的τ-spin observableをHHのfit discriminantへ追加した記述は見つからない。生成器がτ spinを正しくsimulateしていることと、分類器へspin observableを追加することは別である。

[TauSpinner 1201.0117、§2.1、Table 1、Eqs. (3)–(8)](https://arxiv.org/pdf/1201.0117)はW、charged H、neutral H、Z/γ*のlongitudinal polarization/correlationを整理し、τ decay polarimeterを使うevent weightを与える。W由来とcharged-H由来では偏極が逆、neutral scalarとZ/γ*ではlongitudinal correlationの符号が逆。

P≈−1（W）、P≈−0.15（Z pole）という通常の記法はτ⁻のhelicity/charge conventionを明示して用いる。τ⁺のhelicity、polarimeterの符号、各τで定義するk軸を替えると成分の符号が変わる。TauPolarisの**Eq. (18)**ではBk±≈+0.15と記しており、背景説明のP≈−0.15とそのまま符号比較してはいけない。Z/γ*のpolarizationはm_ττやproduction angleにも依存するため、Z poleの定数をHH選択後全域へ一律に使わない。

H→ττは**単一τの平均polarizationが0**でもpair correlationを持つ。TauPolaris **Eqs. (2)–(4)**のbasisではHのCは

\[
C=\begin{pmatrix}\cos2\alpha&\sin2\alpha&0\\-\sin2\alpha&\cos2\alpha&0\\0&0&-1\end{pmatrix},\qquad B_i^\pm=0.
\]

横spin correlationとtt̄のW由来τの単一偏極は異なる識別情報である。fake τには真τのhという定義をそのまま割り当てられない。**single H→ττもHH中のH→ττと同じHのspin構造を持つため、H/Z分離性能だけからsingle-H rejectionを与えることはできない**（spin構造からの推論）。

#### C2. TauPolaris 2608.10961【確定】

[Daniel Winterbottom, Lucas Russell, TauPolaris: reconstructing tau lepton polarimetric vectors with conditional normalizing flows](https://arxiv.org/html/2608.10961)：

- **§III–IV、Table II**：visible decay products、MET、IP/SVをconditionとして、neutrino conditional densityを学び、そこからτ momentumとhを構成する。τhτhは52 inputs。simulationは生成事象にCMSに対応したparameterized detector smearing、track/IP/SV reconstructionを与える。ATLAS full simulationではない。PV resolution x/y=5 μm、z=29 μm（§III）。
- **§IV、Figure 6**：spin-variable regressionのIQR resolutionを、MAP推定がstand-alone Transformerより約40%、posterior samplingより約20%縮める。h全体の一つの角度resolutionやAUCを示す値ではない。
- **§VI、Table III**：φ_CP asymmetryのcombined A_totalはapproximate 0.0529、Transformer 0.0381、TauPolaris 0.0624。従来approximate法に対して18%、Transformerに対して64%のCP sensitivity改善。ρρは0.080→0.093（+16%）、**3π±0π⁰–3π±0π⁰**は0.058→0.109（+88%）。これらはCP測定の指標。
- **§VII、Eq. (18)、Figure 12**：Z poleでC≈diag(0.5,−0.5,1)、B±≈(0,0,0.15)。H/Z識別にcosθn⁺cosθn⁻−cosθr⁺cosθr⁻、cosθk⁺cosθk⁻、cosθk⁺、cosθk⁻の4変数を示す。τhτhの1π±0π⁰、1π±1π⁰、3π±0π⁰を含み、longitudinal productが最も強い分離を示すと記す。
- abstract／§VにはHL-LHCでH→ττのentanglementを少なくとも**4.3σ**で識別するfeasibility resultもある。これはHH productionのdiscovery significanceではない。
- **H/ZのROC、AUC、背景rejection factor、H→ττ discovery gain、HH→bbττ gainは掲載されていない。** Figure 12はnormalized distributions。HとZの質量を125 GeVに揃えたclassification benchmarkも掲載していない。CPの18%をHH discovery significanceの18%と読み替えない。

#### C3. 実験のH→ττ CP解析とpolarimeter性能【確定】

|実験・一次資料|使用した再構成／性能|CP結果（measurement全体でありh angular resolutionではない）|
|---|---|---|
|[CMS 2110.04836](https://arxiv.org/pdf/2110.04836)、§6.1–6.6|IP法、neutral-pion（ρ）法、combined法。a₁³pr–a₁³prのみSV＋τ momentum解＋a₁ currentからpolarimetric vector。§6.4のsimulationでpolarimetric法はneutral-pion法の約2倍のCP-even/odd resolving power。§6.5：IP significance cutsによりneutral-pion法は約2倍のeventへ適用可能|137 fb⁻¹、α_Hττ=−1°±19°、expected 0°±21°。後続CMS paperはRun-2を138 fb⁻¹として記述|
|[ATLAS 2212.05833](https://arxiv.org/pdf/2212.05833)、§3.1–3.4、§5.1、Tables 4–5|φ*_CP、IP、ρ、combined、a₁を含むdecay-planeの構成。τ decay-mode classifierのefficiency/purityは1p0n/1p1nで約80%/70%–80%、3p0nで>90%/90%（§5.1）。これはh regressionのresolutionではない|139 fb⁻¹、φ_τ=9°±16°、expected 0°±28°、pure CP-odd observed rejection 3.4σ（abstract、§8）|
|[CMS 2606.03510](https://arxiv.org/pdf/2606.03510)、§4–5、§12、Figures 4–5|Run-3 PVをbeamspot constraintでrefit、IP/neutral-pion、a₁³prのSVを使うpolarimetric/hybrid法。decay-mode purityはπ/ρ≈80%、a₁¹pr≈70%、a₁³pr≈90%、旧手法より約10%改善（§4）。Table 1と§5に定義|62.4 fb⁻¹のRun 3：expected σ(α)=19°、pure CP-odd expected rejection 2.8σ。Run 2+3：observed 7°±16°、expected 0°±14°（§12.1–12.2）|

CMS 2606.03510のRun-3 expected CP-odd rejectionはτhτh 2.4σ、τμτh 1.3σ、τeτh 0.7σ（§12.1）。このchannel sensitivityをHH channel sensitivityに流用することはできない。

ATLASの「2025 CP」は、[2506.19395](https://arxiv.org/abs/2506.19395)のVBF H→ττにおけるHVV interactionのCP研究と、τ Yukawa CP測定を区別する。ATLAS最新のτ Yukawa φ_τ測定として一次本文を確認できたのは2212.05833。2025年の別のτ Yukawa更新値は本調査では確認できなかった。h vectorのgenerator-truthに対する全mode共通resolution、cosine、H/Z AUCの実験測定値も上記CP papersには得られなかった。

#### C4. CMS HH分類器・DeepTau・ParT【確定／未確定】

[DeepTau 2201.08458、§4.1.1–4.1.2、Table 2](https://arxiv.org/pdf/2201.08458)はPF constituentsのp_T/η/φ/charge、track PV/SV/qualityなどのlow-level inputsと、τ four-momentum、multiplicity、isolation、leading trackとPVのcompatibility、multiprong SVなど**47 high-level inputs**を使用する。§4.2のtraining targetはτh／jet／electron／muonの4クラス。**τ substructureとlifetime/IP情報は既にτ IDに使われている。** しかしexplicit h推定やH/Z pair-spin分類を目的としたtrainingではない。

したがって「CMSにはspinに関係する情報が全く入っていない」という主張は強すぎる。一方、DeepTauを使うことが、最適pair-spin observableをHH classifierで既に使い切ったという証拠にもならない（入力とtraining targetからの推論）。

最新CMS HIG-25-008はTable 1でDeepTau/Boosted DeepTauの使用を確認した。ParT/UParTは[CMS-DP-2025-073](https://cds.cern.ch/record/2946445/files/)というtau reconstruction/ID比較資料があるが、CMS public-resultsの推測したDP2025_073.pdfは404、[CDS本文PDF](https://cds.cern.ch/record/2946445/files/DP2025_073.pdf)もbot確認HTML。本文未取得のため、**ParTの完全なfeature listやHIG-25-008への採用を確定していない**。ParTというarchitecture名だけからspin入力の有無を判定しない。

#### C5. 古典的提案【確定と全文未取得】

- **Bullock–Hagiwara–Martin、Nucl. Phys. B395 (1993) 499–533、DOI 10.1016/0550-3213(93)90045-Q**：[publisher](https://www.sciencedirect.com/science/article/pii/055032139390045Q)、[INSPIRE 336042](https://inspirehep.net/literature/336042)。τ→πν、ρν、a₁ν、ℓννのenergy distributionsからpolarizationを測り、charged/neutral Higgsを識別し、Z→ττのpolarization/correlationを評価する研究。INSPIREの一次publisher abstractと書誌を取得。arXiv番号はmetadataにない。publisher全文は403 Forbidden、INSPIREにPDF resourceなしのため、具体的cut効率や表番号を本文由来としては提示できない。
- **Guchait–Roy、[0808.0438](https://arxiv.org/pdf/0808.0438)**：Using Tau Polarization for Charged Higgs Boson and SUSY searches at LHC。abstract／§2–3で1-prongのcharged-track momentum fraction **R>0.8**を用い、Pτ=+1のsignalを保ちPτ=−1のW背景とfakeを抑える提案。SM HHのH→ττには両単一τがP=+1という前提は適用されない。
- **Ali et al.、[1103.1827](https://arxiv.org/pdf/1103.1827)**：Improved sensitivity to charged Higgs searches in Top quark decays t→bH⁺→b(τ⁺ντ) at the LHC using τ polarisation and multivariate techniques。abstract、§2–4でρ→π±π⁰のenergy/momentum fractionなどのpolarization variablesをBDTへ入れ、W由来tt̄背景に対するH±を識別。**偏極＋MVAの先例だがneutral SM HHではない。**
- **TauSpinner [1201.0117](https://arxiv.org/pdf/1201.0117)、[1406.1647](https://arxiv.org/abs/1406.1647)**：longitudinal/transverse spin correlationsの生成・reweighting。H/Z/tt̄のpolarizationを正しくモデル化する基盤であり、reconstructed hをHH fitへ足した感度結果ではない。

- **Kalinowski et al.、[1604.00964](https://arxiv.org/pdf/1604.00964)、§5.1–5.2、Tables 4–7**：ττ＋high-p_T jetsで2→2／2→4 matrix elementsを用いたpolarizationを比較。Table 7のZ pole、effective EW scheme EWSH=4ではPτ=**−0.1488±0.0008／−0.1486±0.0008**。同じ表のon-shell sin²θ_W=0.222246では約−0.214となる。HH選択へPτ≈−0.15を移植するときにEW schemeとphase spaceを合わせる必要がある。Table 6のm_ττ,m_jj>120 GeV、VBF-like選択で−0.5026／−0.5184／−0.5110となる例はEWSH=1であり、Z poleの普遍値ではない。
- **Boos et al.、[hep-ph/0507100](https://arxiv.org/pdf/hep-ph/0507100)、Figures 2–3、Concluding remarks**：ILCのt→bH±とt→bWのτ polarization差からπ energy spectrumをfitし、H± massを**0.5–1 GeV**で推定する理論水準研究。detector/systematic effectsは含まない。HHのprofile-likelihood改善率を示してはいない。
- **Jeans–Wilson、[1804.01241](https://arxiv.org/abs/1804.01241)、§IV.C/E/F、§V、Table IV**：ILD full simulationでτ momentum／polarimeterとIPを再構成し、NNでbackground purityをカテゴリ化したCP fit。2 ab⁻¹、250 GeVでψ_CP精度**75 mrad（4.3°）**（abstract、§V）。§IV.EのNN inputsはmass、event energy、recoilとτ周辺energy／visible massで、spin polarimetersは§IV.FのCP observableへ使う。再構成polarimeter＋MVA categories＋尤度の先例だが、e⁺e⁻→ZHのCP測定である。

**特に近いH/Z分類の先例**：[Michaela Roggendorf, BONN-IB-2014-03, Studies of Tau-Lepton Polarisation in Decays of Higgs and Z Bosons with the ATLAS Experiment（2014修士論文）](https://inspirehep.net/files/732c0f9e050fda120c785eadebb34c27)。**§5.4.3、§6.1、§6.4.3、Figure 6.18、Table 6.11**を実際に取得・確認した。8 TeVのATLAS MC、H(125 GeV)とZで、MMC neutrino reconstructionとπ⁰ reconstructionからTauSpinnerのpolarimetric vectorの**z成分H_Z**を構成し、ρρでx、Υ、H_Zの組をTMVA BDTへ入れる。Table 6.11の全入力のseparationはtruth **0.141**、reconstruction **0.055**、x＋Υだけのreconstructionは**0.053**。ここでseparationは**Eq. (6.1)**のnormalized-distribution指標（½Σ(A−B)²/(A+B)）であり、AUCや実験discovery σではない。全hのML regression、IP/SVをconditionとする推定、HH fit、HL-LHC projectionは扱わない。**「reco polarimeterをH/ZのMVAへ入れる」こと自体には明確な先例がある。**

Cの未確定事項：検索語“HH bbtautau tau polarization”“Higgs pair tau spin”“di-Higgs tau polarimetric vector”とその綴り違い、arXiv/INSPIRE・ATLAS/CMS公開ページを調べたが、HHのreco h追加によるHL-LHC感度を掲載した一致文献を取得できなかった。検索にはastroのHH objectsなど無関係な結果もあり、それらを根拠に含めていない。

### D. 新規性の判定

**確認した公開文献の範囲では、再構成水準hをHH→bbττのevent MVAへ追加し、HL-LHCのprofile likelihoodでSM HH discovery sensitivityを出した研究は見つからない。** 未公開実験noteや未索引のthesisまで不存在を証明したものではない。

|最近接研究|既に示したこと|今回の研究との相違|
|---|---|---|
|TauPolaris 2608.10961|visible＋MET＋IP/SVからneutrino posterior、h、H/Z spin variables、CP sensitivity|CMSに対応するsmearing、HH event classifier/背景fitなし。今回のATLAS full simulation、直接h regression、同質量H/Z benchmark、HH fit追加効果は別の評価|
|BONN-IB-2014-03（Roggendorf）|ATLAS MCでreco H_Zをx、Υと共にH/Z BDTへ追加|MMCベースのz成分、8 TeV。全hの回帰やHHのHL-LHC fitを扱わない|
|CMS H→ττ CP 2110.04836 / 2606.03510|実データでIP/SV、polarimetric vector、CP angle fit|目的はτ Yukawa CP測定。HH discoveryのadditional spin discriminationを定量化していない|
|1103.1827、0808.0438|τ polarizationでW/tt̄を抑制、MVAへの追加|H±／SUSY、単一τのlongitudinal polarization。SM neutral H pairのspin-correlation signalと背景組成が異なる|
|ATLAS 2607.26879|Transformer＋GN2＋trigger更新、HH fit|Table 2にexplicit τ-spin/IP/SV inputなし。hの追加感度を測る比較対象|

よって「hをIP/SVから再構成する」「spinでH/Zを分ける」だけを新規性の中心にするのは難しい。**現在のHH分類器と同一選択・同一systematic scenarioで、追加spin情報がどの背景をどのbinで減らし、profile likelihoodをどれだけ改善するか**が、最近接研究でまだ掲載されていない結果になるという判定である。

公開情報で今回埋められたのは、4.3σの出典、投影元の識別、外挿規則、legacy全binの合算process組成、最新のSR総収量・選択・入力・主要NPである。**最新／投影元のbin別workspaceと、追加hのprocess別かつ旧score条件付き分布は得られなかった。** このため、ユーザー提示AUC 0.626（exact h 0.72–0.74）は背景各processのbin migrationやcorrelated NPを決めず、最新HHに対する一つの確定したσ改善率へ換算できない。

尤度として必要な内容は、例えば

\[
\mathcal L(\mu,\theta)=
\prod_{c,i,j}{\rm Pois}\!\left(n_{cij}\mid
\mu s_{cij}(\theta)+\sum_p b_{pcij}(\theta)\right)
\prod_k f_k(\theta_k)
\]

におけるc=カテゴリ、i=既存score bin、j=追加spin bin、p=background processのtemplatesとNP response、CRとの相関である。公開されたpost-fit stackはこの全情報を含まない。これは「調べる必要がある」という未着手の結論ではなく、下記URLの本文・補助表・HEPData resourceまで取得して確認した公開範囲の限界である。

## 3. カテゴリ×score binのprocess収量または取得先

### 3.1 最新ATLAS：全score binの積分収量

出典：[Auxiliary Table 01](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/tabaux_01.pdf)（τhadτhad）、[Auxiliary Table 02](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/tabaux_02.pdf)（τlepτhad）。**combined μ_HH unconditional fit後**の数値。HHもpost-fit yieldであり、SM-normalized expected yieldではない。図のfit valueは2.59、本文では2.6と丸める。表の表示値を最終binの値へ置換していない。以下はcentral valuesで、誤差は取得PDFにある。

|channel / region / run|score bin|ggF HH|VBF HH|tt̄ true τ|tW|jet fake（Top,W）|lepton fake|MJ fake|Z+HF|single H|Other|Total B|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|τhadτhad Hi R2|全bin合計|16.3|0.192|1343|96|1338|713|1550|753|57|150|5995|
|τhadτhad Hi R3|全bin合計|7.4|0.083|563|34|855|329|1153|346|26.7|76|3382|
|τhadτhad Lo R2|全bin合計|1.74|0.033|722|21|1352|497|970|515|23.2|78|4182|
|τhadτhad Lo R3|全bin合計|2.10|0.035|666|20|2581|564|5670|379|21.9|88|9992|
|τhadτhad VBF R2|全bin合計|0.30|0.34|206|6.8|194|93.9|382|52.4|3.8|24|963|
|τhadτhad VBF R3|全bin合計|0.171|0.179|114|2.6|269|69|973|32.3|2.58|18.6|1481|
|τlepτhad Hi R2|全bin合計|14.8|0.168|21859|1360|16990|507|含む|739|91|267|41810|
|τlepτhad Hi R3|全bin合計|6.1|0.069|8847|460|9270|204.4|含む|310|40.5|125|19250|
|τlepτhad Lo R2|全bin合計|2.7|0.052|29485|1210|32900|649|含む|1400|81.8|329|66010|
|τlepτhad Lo R3|全bin合計|1.07|0.0207|11640|390|17310|261|含む|606|34.9|160|30400|
|τlepτhad VBF R2|全bin合計|0.29|0.38|9804|460|6770|218|含む|193|10.9|81.7|17540|
|τlepτhad VBF R3|全bin合計|0.121|0.155|4066|160|3940|85.7|含む|82|5.14|37.1|8378|

τlepτhadのMJはfake-τhad estimateに含まれ独立収量なし。各componentは丸められ、totalは丸め前に計算される。single HをggF/VBF/ZH/ttHに分解した表もない。

各score binの**図の取得先**は以下。数字のhistogram JSON/YAMLは未取得だが、PDFは全て保存した。最終classifier binは順序番号として図に示され、連続score boundaryは公開図だけから確定できない。

|run / channel / SR|公開図・直接PDF|
|---|---|
|Run 2 τhadτhad Lo|[Figure 7a](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_07a.pdf)|
|Run 2 τlepτhad Lo|[Figure 7b](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_07b.pdf)|
|Run 2 τhadτhad Hi|[Figure 7c](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_07c.pdf)|
|Run 2 τlepτhad Hi|[Figure 7d](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_07d.pdf)|
|Run 2 τhadτhad VBF|[Figure 7e](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_07e.pdf)|
|Run 2 τlepτhad VBF|[Figure 7f](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_07f.pdf)|
|Run 3 τhadτhad Lo|[Figure 8a](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_08a.pdf)|
|Run 3 τlepτhad Lo|[Figure 8b](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_08b.pdf)|
|Run 3 τhadτhad Hi|[Figure 8c](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_08c.pdf)|
|Run 3 τlepτhad Hi|[Figure 8d](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_08d.pdf)|
|Run 3 τhadτhad VBF|[Figure 8e](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_08e.pdf)|
|Run 3 τlepτhad VBF|[Figure 8f](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_08f.pdf)|

### 3.2 legacy：本文Table 5の最終bin

出典：[2209.10910、Table 5](https://arxiv.org/pdf/2209.10910)および[HEPData Table 5](https://www.hepdata.net/download/table/ins2155171/Table%205/json)。上の不整合により、SM signalと精密な最終bin組成には本文値を優先する。

|process|τhadτhad 最終bin|τlepτhad SLT 最終bin|τlepτhad LTT 最終bin|
|---|---:|---:|---:|
|SM ggF HH|1.58±0.27|0.77±0.13|0.25±0.05|
|SM VBF HH|0.0227±0.0019|0.0075±0.0007|0.00455±0.00035|
|tt̄|0.5±0.1|0.44±0.12|1.76±0.32|
|Single top-quark|0.47±0.20|1.2±0.7|0.61±0.35|
|Z+HF|2.3±0.4|1.55±0.33|1.7±0.4|
|Combined fakes|—|1.09±0.22|0.8±0.5|
|Multi-jet fakes|0.47±0.17|—|—|
|tt̄ fakes|0.29±0.07|—|—|
|Single Higgs boson|1.7±0.5|1.10±0.28|0.4±0.1|
|Other backgrounds|0.4±0.1|0.48±0.09|0.33±0.07|
|Total B|6.1±0.8|6±1|6±1|
|Data|8|7|7|

### 3.3 legacy：全score binのHEPData原数値

出典：[Figure 8a JSON](https://www.hepdata.net/download/table/ins2155171/Figure%208a/json)、[8b JSON](https://www.hepdata.net/download/table/ins2155171/Figure%208b/json)、[8c JSON](https://www.hepdata.net/download/table/ins2155171/Figure%208c/json)。CSV：[data/legacy_score_bins_raw.csv](data/legacy_score_bins_raw.csv)。数値は表示を6 significant digitsへ丸め、元JSON/YAMLの全桁を保持した。**HH列はSM yieldではなくraw expected-limit normalization、LTT Total B列は不整合を含む。** Topはtt̄＋single topをまとめた列。single Hはproduction modeに分かれない。

|カテゴリ|bin（小→大）|HH（expected limit、原データ）|Top|MJ fake|tt̄ fake／combined fake|Z+HF|single H|Other|Total B（原データ）|Data|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|τhadτhad|1|0.0214865|1755.49|398.554|925.825|161.996|5.427|40.158|3287.45|3252|
|τhadτhad|2|0.0605508|588.698|256.814|412.877|284.173|4.908|31.909|1579.38|1569|
|τhadτhad|3|0.153826|346.272|212.161|289.574|326.626|6.144|30.814|1211.59|1220|
|τhadτhad|4|0.280116|199.299|133.255|169.673|242.489|6.193|24.983|775.892|808|
|τhadτhad|5|0.493388|133.533|110.045|104.498|194.053|6.475|19.257|567.86|541|
|τhadτhad|6|0.627489|74.93|52.424|59.218|104.123|5.89|15.48|312.066|326|
|τhadτhad|7|0.780567|51.875|38.606|40.19|79.841|5.089|10.291|225.892|229|
|τhadτhad|8|0.988037|33.378|28.801|27.84|54.427|4.996|7.482|156.925|141|
|τhadτhad|9|1.32327|24.93|20.648|21.381|41.266|5.369|3.853|117.447|122|
|τhadτhad|10|1.53224|16.939|13.579|12.357|27.005|4.321|4.239|78.439|81|
|τhadτhad|11|1.91136|13.364|3.392|8.577|16.313|3.802|3.369|48.817|45|
|τhadτhad|12|2.65127|4.87|4.071|3.7|10.888|3.389|2.167|29.085|23|
|τhadτhad|13|3.43372|2.812|1.477|2.58|6.031|2.601|1.123|16.625|15|
|τhadτhad|14|5.77068|0.983|0.466|0.286|2.261|1.721|0.414|6.133|8|
|τlepτhad SLT|1|0.0022801|9291.1|—|2134.18|2.379|9.79|19.172|11456.6|11429|
|τlepτhad SLT|2|0.0096731|13709.3|—|6826.1|27.092|17.811|64.433|20644.7|20671|
|τlepτhad SLT|3|0.0448086|11436.5|—|7854.38|67.947|18.238|124.657|19501.7|19564|
|τlepτhad SLT|4|0.184205|10203.9|—|7424.93|169.287|19.96|240.761|18058.9|17898|
|τlepτhad SLT|5|0.628936|8260.65|—|5501.01|416.399|21.753|326.861|14526.7|14613|
|τlepτhad SLT|6|1.37157|4660.3|—|2754.11|394.379|17.42|242.972|8069.19|8151|
|τlepτhad SLT|7|1.89305|2015.88|—|1078.3|232.236|11.836|135.771|3474.02|3469|
|τlepτhad SLT|8|2.11801|812.429|—|433.584|133.266|8.201|59.645|1447.13|1381|
|τlepτhad SLT|9|2.22475|366.779|—|186.774|75.203|6.336|29.76|664.852|697|
|τlepτhad SLT|10|2.29224|162.162|—|67.567|49.165|4.982|23.519|307.395|307|
|τlepτhad SLT|11|2.3421|67.347|—|31.269|25.961|3.936|13.718|142.23|148|
|τlepτhad SLT|12|2.30179|33.274|—|13.305|15.293|3.054|6.857|71.783|68|
|τlepτhad SLT|13|2.28049|14.221|—|3.725|8.144|2.411|3.251|31.752|38|
|τlepτhad SLT|14|2.29116|6.514|—|1.264|4.484|1.615|1.493|15.37|15|
|τlepτhad SLT|15|2.90416|1.658|—|1.089|1.549|1.099|0.484|5.879|7|
|τlepτhad LTT|1|0.00089732|340.308|—|145.11|2.379|0.588|1.946|491.535|480|
|τlepτhad LTT|2|0.00448785|753.237|—|397.34|27.092|1.934|8.056|1181.47|1212|
|τlepτhad LTT|3|0.0272422|732.164|—|356.306|67.947|2.238|12.817|1160.77|1112|
|τlepτhad LTT|4|0.0658325|733.092|—|293.764|169.287|2.972|23.584|1132.7|1149|
|τlepτhad LTT|5|0.17134|582.775|—|186.664|416.399|3.044|16.262|896.684|914|
|τlepτhad LTT|6|0.289272|390.373|—|122.911|394.379|2.688|17.299|620.387|627|
|τlepτhad LTT|7|0.388549|226.084|—|71.863|232.236|2.317|15.423|378.908|363|
|τlepτhad LTT|8|0.471631|122.542|—|28.731|133.266|1.883|8.437|195.098|203|
|τlepτhad LTT|9|0.502113|68.779|—|16.976|75.203|1.446|7.607|121.114|126|
|τlepτhad LTT|10|0.525195|38.791|—|11.113|49.165|1.011|2.912|72.138|58|
|τlepτhad LTT|11|0.529409|25.51|—|3.482|25.961|0.86|2.638|45.709|62|
|τlepτhad LTT|12|0.530352|14.262|—|4.082|15.293|0.663|1.701|27.97|20|
|τlepτhad LTT|13|0.528171|7.545|—|1.304|8.144|0.542|0.721|16.145|12|
|τlepτhad LTT|14|0.523014|4.178|—|1.563|4.484|0.392|0.391|9.806|6|
|τlepτhad LTT|15|0.956815|2.371|—|0.795|1.549|0.402|0.333|2.371|7|


### 3.4 投影元2404.12660のscore分布の取得先

9カテゴリの図は**Figure 6a–i**。公表HEPData ins2779337にはその数値histogramがなく、公開表も入力一覧／efficiencyである。図のprocess groupingはTop-quark（tt̄＋single top）、jet→τhad fakes、τhadτhadのみ別のtt̄ fakes、Z+HF、Single Higgs、Other。single-H production modesは分かれない。

|channel|ggF Lo（m_HH<350 GeV）|ggF Hi（m_HH≥350 GeV）|VBF|
|---|---|---|---|
|τhadτhad|[Figure 6a](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HDBS-2019-27/fig_06a.pdf)|[Figure 6b（保存済み）](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HDBS-2019-27/fig_06b.pdf)|[Figure 6c](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HDBS-2019-27/fig_06c.pdf)|
|τlepτhad SLT|[Figure 6d](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HDBS-2019-27/fig_06d.pdf)|[Figure 6e](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HDBS-2019-27/fig_06e.pdf)|[Figure 6f](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HDBS-2019-27/fig_06f.pdf)|
|τlepτhad LTT|[Figure 6g](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HDBS-2019-27/fig_06g.pdf)|[Figure 6h](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HDBS-2019-27/fig_06h.pdf)|[Figure 6i](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HDBS-2019-27/fig_06i.pdf)|

## 4. 機械取得URL、保存ファイル、取得失敗

### 4.1 数値データと公開modelの確認結果

HEPDataへの最初のアクセスは次の形式で行った。HTTP statusだけでなくContent-Typeと本文を検査した。

```sh
curl -sL -H "Accept: application/json"   "https://www.hepdata.net/record/ins2155171?format=json"
curl -sL -H "Accept: application/json"   "https://www.hepdata.net/search/?q=2607.26879&format=json"
```

|対象|URL／table id|保存ファイル|中身／model可否|
|---|---|---|---|
|legacy record|[ins2155171 JSON](https://www.hepdata.net/record/ins2155171?format=json)、record 130794 v1|[hepdata_legacy_record.json](data/hepdata_legacy_record.json)|51 tablesの目録、各download URL、resources|
|legacy Figure 8a|[JSON](https://www.hepdata.net/download/table/ins2155171/Figure%208a/json)、id 1380037|[hepdata_legacy_fig8a.json](data/hepdata_legacy_fig8a.json)|τhadτhad、14 bins、process yields|
|legacy Figure 8b|[JSON](https://www.hepdata.net/download/table/ins2155171/Figure%208b/json)、id 1380038|[hepdata_legacy_fig8b.json](data/hepdata_legacy_fig8b.json)|τlepτhad SLT、15 bins、process yields|
|legacy Figure 8c|[JSON](https://www.hepdata.net/download/table/ins2155171/Figure%208c/json)、id 1380039|[hepdata_legacy_fig8c.json](data/hepdata_legacy_fig8c.json)|τlepτhad LTT、15 bins、Total B不整合あり|
|legacy Table 5|[JSON](https://www.hepdata.net/download/table/ins2155171/Table%205/json)、id 1380020|[hepdata_legacy_table5.json](data/hepdata_legacy_table5.json)|最終binのsummary yield。descriptionの“last two bins”は本文と不一致|
|legacy raw YAML|[Figure 8a](https://www.hepdata.net/download/table/ins2155171/Figure%208a/yaml)、[8c](https://www.hepdata.net/download/table/ins2155171/Figure%208c/yaml)|[legacy_fig8a.yaml](data/legacy_fig8a.yaml)、[legacy_fig8c.yaml](data/legacy_fig8c.yaml)|JSONをcross-check。ROOTはrecordがdownload URLを提供するが本調査では未取得|
|legacy全submission|[YAML archive](https://www.hepdata.net/download/submission/ins2155171/1/yaml)|[hepdata_legacy_record.yaml.tar.gz](data/hepdata_legacy_record.yaml.tar.gz)|全51表。pyhf/HistFactory JSONなし|
|投影元2404.12660 record|[ins2779337 JSON](https://www.hepdata.net/record/ins2779337?format=json)、record 151276 v1|[hepdata_projection_base_record.json](data/hepdata_projection_base_record.json)|32 tables。yield/workspaceなし、scanあり|
|投影元expected κλ scan|[Combined −2lnΛ vs κλ exp JSON](https://www.hepdata.net/download/table/ins2779337/Combined%20%24-2%5Cln%5CLambda%24%20vs%20%24%5Ckappa_%7B%5Clambda%7D%24%20exp/json)、id 1714476|[hepdata_projection_base_kl_expected.json](data/hepdata_projection_base_kl_expected.json)|Figure 8aのscan。NP responseやprocess templatesを持たない|
|投影元全submission|[YAML archive](https://www.hepdata.net/download/submission/ins2779337/1/yaml)|[hepdata_projection_base.yaml.tar.gz](data/hepdata_projection_base.yaml.tar.gz)|32 data YAML＋submission。modelなし|
|最新ATLAS record試行|[ins3184980 JSON](https://www.hepdata.net/record/ins3184980?format=json)|hepdata_latest_record.json|404 HTML。拡張子はJSONだが数値JSONではない|
|最新CMS record検索|[HIG-25-008 search JSON](https://www.hepdata.net/search/?q=HIG-25-008&format=json)|hepdata_cms_latest_search.json|対象論文のrecordを同定できず。public pageのHEPData IDはplaceholder|
|公開pyhf/HistFactory model|最新、投影元、legacyの公開ページ／HEPData resourcesを確認|該当ファイルなし|取得できたscanとpost-fit yieldsはmodelではない|

archiveのファイル名は取得時の名称を保った。Content-Typeはapplication/x-tarであり、`.tar.gz`というsuffixからcompressionを決めず、tarfileで実体を開いて確認した。archiveから取り出した目録は[hepdata_submission.yaml](data/hepdata_submission.yaml)と[hepdata_projection_base_submission.yaml](data/hepdata_projection_base_submission.yaml)。[legacy_archive_fig08c.yaml](data/legacy_archive_fig08c.yaml)はarchive中のfig8c_PNN.yamlの複製で、LTT列の検証用。[legacy_score_bins_raw.csv](data/legacy_score_bins_raw.csv)は44 rowsを元JSONから整理した派生データで、物理的な再規格化は加えていない。

### 4.2 保存した一次資料・補助PDF・metadataの一覧

以下のファイルは全て`data/`内。PDFはmagic bytesを検査し、同stemの`.txt`に`pdftotext -layout`の抽出を保存した。`.txt`は閲覧用の派生物で、数値の正本はPDF/JSON/YAML。公開ページの一部にも同stemの`.txt`を保存した。各取得の最終URL、bytes、HTTP statusは[fetch_manifest.json](data/fetch_manifest.json)にある。

|保存ファイル|取得URL|内容|
|---|---|---|
|[atlas_latest.html](data/atlas_latest.html)|[取得URL](https://arxiv.org/html/2607.26879)|2607.26879のarXiv HTML本文|
|[atlas_latest.pdf](data/atlas_latest.pdf)|[取得URL](https://arxiv.org/pdf/2607.26879)|2607.26879本文、196 fb⁻¹、Transformer、選択／fit／系統|
|[taupolaris.html](data/taupolaris.html)|[取得URL](https://arxiv.org/html/2608.10961)|TauPolaris arXiv HTML本文|
|[taupolaris.pdf](data/taupolaris.pdf)|[取得URL](https://arxiv.org/pdf/2608.10961)|2608.10961本文、h再構成、CP、H/Z spin variables|
|[highlights.pdf](data/highlights.pdf)|[取得URL](https://arxiv.org/pdf/2504.00672)|2504.00672共同投影、Table 2、ATLAS+CMS仮定|
|[legacy.pdf](data/legacy.pdf)|[取得URL](https://arxiv.org/pdf/2209.10910)|2209.10910 v2本文、最終bin Table 5|
|[legacy.html](data/legacy.html)|[取得URL](https://arxiv.org/html/2209.10910)|2209.10910 arXiv HTML本文|
|[atlas_bbtautau_2024.pdf](data/atlas_bbtautau_2024.pdf)|[取得URL](https://arxiv.org/pdf/2404.12660)|2404.12660本文、HL-LHC bbττの投影元|
|[hepdata_search_latest.json](data/hepdata_search_latest.json)|[取得URL](https://www.hepdata.net/search/?q=2607.26879&format=json)|HEPData検索応答。対象recordの同定に使い、無関係なhitも保持|
|[hepdata_search_legacy.json](data/hepdata_search_legacy.json)|[取得URL](https://www.hepdata.net/search/?q=2209.10910&format=json)|HEPData検索応答。対象recordの同定に使い、無関係なhitも保持|
|[atlas_projection_page.html](data/atlas_projection_page.html)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PUBNOTES/ATL-PHYS-PUB-2025-006/)|2025-006の公開figures/tables目録|
|[cms_latest_page.html](data/cms_latest_page.html)|[取得URL](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/HIG-25-008/)|HIG-25-008公開Summary、全figures/tablesへのリンク|
|[inspire_latest.json](data/inspire_latest.json)|[取得URL](https://inspirehep.net/api/literature?q=arxiv:2607.26879)|INSPIRE書誌／external identifiers／PDFリンク。検索結果の該当有無も保持|
|[inspire_legacy.json](data/inspire_legacy.json)|[取得URL](https://inspirehep.net/api/literature?q=arxiv:2209.10910)|INSPIRE書誌／external identifiers／PDFリンク。検索結果の該当有無も保持|
|[hepdata_legacy_record.json](data/hepdata_legacy_record.json)|[取得URL](https://www.hepdata.net/record/ins2155171?format=json)|HEPData数値／目録（§4.1）|
|[inspire_atlas_projection.json](data/inspire_atlas_projection.json)|[取得URL](https://inspirehep.net/api/literature/2898222)|INSPIRE書誌／external identifiers／PDFリンク。検索結果の該当有無も保持|
|[cms_legacy.pdf](data/cms_legacy.pdf)|[取得URL](https://arxiv.org/pdf/2206.09401)|CMS 2206.09401本文、Run-2 bbττ/DNN|
|[cms_cp_2021.pdf](data/cms_cp_2021.pdf)|[取得URL](https://arxiv.org/pdf/2110.04836)|CMS 2110.04836本文、IP/ρ/a₁ polarimetric法|
|[cms_cp_2026.pdf](data/cms_cp_2026.pdf)|[取得URL](https://arxiv.org/pdf/2606.03510)|CMS 2606.03510本文、Run-3 CPとRun2+3結果|
|[atlas_projection_full.pdf](data/atlas_projection_full.pdf)|[取得URL](https://inspirehep.net/files/5cabc0e66dede649b8f6e37147e87a82)|ATL-PHYS-PUB-2025-006全文、4.26σ正本|
|[atlas_legacy_correct_page.html](data/atlas_legacy_correct_page.html)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HDBS-2018-40/)|HDBS-2018-40入口のmeta-refresh HTML|
|[cms_note_inspire.json](data/cms_note_inspire.json)|[取得URL](https://inspirehep.net/api/literature?q=reportnumber:CMS-NOTE-2025-006)|CMS note report-number検索応答（該当なし）|
|[cms_latest_table1.pdf](data/cms_latest_table1.pdf)|[取得URL](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/HIG-25-008/CMS-PAS-HIG-25-008_Table_001.pdf)|CMS Table 1、resolved/boosted τの選択|
|[cms_latest_table2.pdf](data/cms_latest_table2.pdf)|[取得URL](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/HIG-25-008/CMS-PAS-HIG-25-008_Table_002.pdf)|CMS Table 2、relative uncertainty contributions|
|[cms_latest_table3.pdf](data/cms_latest_table3.pdf)|[取得URL](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/HIG-25-008/CMS-PAS-HIG-25-008_Table_003.pdf)|CMS Table 3、Run2/3/combined μ limits|
|[atlas_cp.pdf](data/atlas_cp.pdf)|[取得URL](https://arxiv.org/pdf/2212.05833)|ATLAS 2212.05833本文、τ Yukawa CP|
|[hepdata_legacy_table5.json](data/hepdata_legacy_table5.json)|[取得URL](https://www.hepdata.net/download/table/ins2155171/Table%205/json)|HEPData数値／目録（§4.1）|
|[hepdata_legacy_fig8a.json](data/hepdata_legacy_fig8a.json)|[取得URL](https://www.hepdata.net/download/table/ins2155171/Figure%208a/json)|HEPData数値／目録（§4.1）|
|[hepdata_legacy_fig8b.json](data/hepdata_legacy_fig8b.json)|[取得URL](https://www.hepdata.net/download/table/ins2155171/Figure%208b/json)|HEPData数値／目録（§4.1）|
|[hepdata_legacy_fig8c.json](data/hepdata_legacy_fig8c.json)|[取得URL](https://www.hepdata.net/download/table/ins2155171/Figure%208c/json)|HEPData数値／目録（§4.1）|
|[atlas_latest_page.html](data/atlas_latest_page.html)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/)|HIGP-2024-37公開ページ、全resourceの目録|
|[atlas_legacy_index.html](data/atlas_legacy_index.html)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HDBS-2018-40/index.php)|HDBS-2018-40の実ページ、legacy auxiliary目録|
|[inspire_cms_latest.json](data/inspire_cms_latest.json)|[取得URL](https://inspirehep.net/api/literature?q=reportnumber%20HIG-25-008)|INSPIRE書誌／external identifiers／PDFリンク。検索結果の該当有無も保持|
|[inspire_cms_note_title.json](data/inspire_cms_note_title.json)|[取得URL](https://inspirehep.net/api/literature?q=title:%22Projection%20of%20CMS%20experimental%20reach%20on%20HH%20production%20at%20HL-LHC%22)|INSPIRE書誌／external identifiers／PDFリンク。検索結果の該当有無も保持|
|[latest_tabaux01.pdf](data/latest_tabaux01.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/tabaux_01.pdf)|Auxiliary Table 01、τhadτhad 6 SRの積分収量|
|[latest_tabaux02.pdf](data/latest_tabaux02.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/tabaux_02.pdf)|Auxiliary Table 02、τlepτhad 6 SRの積分収量|
|[latest_tabaux03.pdf](data/latest_tabaux03.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/tabaux_03.pdf)|Auxiliary Table 03、ZH/ZZ系統分解|
|[latest_tabaux04.pdf](data/latest_tabaux04.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/tabaux_04.pdf)|Auxiliary Table 04、VBF signal strength系統分解|
|[legacy_tabaux01.pdf](data/legacy_tabaux01.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HDBS-2018-40/tabaux_01.pdf)|legacy Auxiliary Table 01：τhadτhad selection cutflow/A×ε|
|[legacy_tabaux04.pdf](data/legacy_tabaux04.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HDBS-2018-40/tabaux_04.pdf)|legacy Auxiliary Table 04：3カテゴリの全bin収量（SM HH、背景、Data）|
|[legacy_tabaux09.pdf](data/legacy_tabaux09.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HDBS-2018-40/tabaux_09.pdf)|legacy Auxiliary Table 09：non-resonant/resonant HHのcross-section limits|
|[projection_tab07.pdf](data/projection_tab07.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PUBNOTES/ATL-PHYS-PUB-2025-006/tab_07.pdf)|2025-006 Table 7、κλ 68% CI|
|[inspire_2024.json](data/inspire_2024.json)|[取得URL](https://inspirehep.net/api/literature?q=arxiv:2404.12660)|INSPIRE書誌／external identifiers／PDFリンク。検索結果の該当有無も保持|
|[inspire_projection_bbtautau.json](data/inspire_projection_bbtautau.json)|[取得URL](https://inspirehep.net/api/literature?q=reportnumber:ATL-PHYS-PUB-2025-001)|INSPIRE書誌／external identifiers／PDFリンク。検索結果の該当有無も保持|
|[latest_fig9a.pdf](data/latest_fig9a.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_09a.pdf)|Figure 9a、observed/expected μ_HH upper limit|
|[latest_figaux31.pdf](data/latest_figaux31.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/figaux_31.pdf)|Auxiliary Figure 31、Asimov定義／luminosity scalingごとの期待σ|
|[latest_figaux32.pdf](data/latest_figaux32.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/figaux_32.pdf)|Auxiliary Figure 32、Asimov定義／luminosity scalingごとのexpected limit|
|[cms_latest_fig2.pdf](data/cms_latest_fig2.pdf)|[取得URL](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/HIG-25-008/CMS-PAS-HIG-25-008_Figure_002.pdf)|CMS Figure 2、resolved 2bのpre-fit m_HH分布|
|[cms_latest_fig3.pdf](data/cms_latest_fig3.pdf)|[取得URL](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/HIG-25-008/CMS-PAS-HIG-25-008_Figure_003.pdf)|CMS Figure 3、resolved 2bのpre-fit p_T,HH分布|
|[tauspin_classic.pdf](data/tauspin_classic.pdf)|[取得URL](https://arxiv.org/pdf/1201.0117)|1201.0117本文、TauSpinner longitudinal spin weights|
|[taupol_roy.pdf](data/taupol_roy.pdf)|[取得URL](https://arxiv.org/pdf/0808.0438)|0808.0438本文、R>0.8 polarization cut|
|[atlas_bbgammagamma_projection.pdf](data/atlas_bbgammagamma_projection.pdf)|[取得URL](https://inspirehep.net/files/11275dd9694043e1f9e64d88272750cb)|ATL-PHYS-PUB-2025-001：bbγγの投影。bbττ noteではない|
|[legacy_fig8a.yaml](data/legacy_fig8a.yaml)|[取得URL](https://www.hepdata.net/download/table/ins2155171/Figure%208a/yaml)|HEPData数値／目録（§4.1）|
|[legacy_fig8c.yaml](data/legacy_fig8c.yaml)|[取得URL](https://www.hepdata.net/download/table/ins2155171/Figure%208c/yaml)|HEPData数値／目録（§4.1）|
|[deeptau.pdf](data/deeptau.pdf)|[取得URL](https://arxiv.org/pdf/2201.08458)|2201.08458本文、PF/IP/SV inputs、4-class τ ID|
|[charged_higgs_bdt.pdf](data/charged_higgs_bdt.pdf)|[取得URL](https://arxiv.org/pdf/1103.1827)|1103.1827本文、charged Hのτ polarization＋BDT|
|[inspire_projection2024.json](data/inspire_projection2024.json)|[取得URL](https://inspirehep.net/api/literature?q=reportnumber:ATL-PHYS-PUB-2024-016)|INSPIRE書誌／external identifiers／PDFリンク。検索結果の該当有無も保持|
|[classic_bhm_inspire.json](data/classic_bhm_inspire.json)|[取得URL](https://inspirehep.net/api/literature?q=doi:10.1016/0550-3213(93)90045-Q)|INSPIRE書誌／external identifiers／PDFリンク。検索結果の該当有無も保持|
|[cms_latest_fig4.pdf](data/cms_latest_fig4.pdf)|[取得URL](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/HIG-25-008/CMS-PAS-HIG-25-008_Figure_004.pdf)|CMS Figure 4、2024カテゴリ別DNN score stacks|
|[latest_figaux25a.pdf](data/latest_figaux25a.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/figaux_25a.pdf)|Auxiliary Figure 25a、κλ–κ2V contour|
|[latest_fig10.pdf](data/latest_fig10.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_10.pdf)|Figure 10、log10(S/B)で再集約したpost-fit分布|
|[hepdata_cms_latest_search.json](data/hepdata_cms_latest_search.json)|[取得URL](https://www.hepdata.net/search/?q=HIG-25-008&format=json)|HEPData検索応答。対象recordの同定に使い、無関係なhitも保持|
|[latest_fig07a.pdf](data/latest_fig07a.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_07a.pdf)|最新ATLAS 12 SRのpost-fit score分布（§3.1にカテゴリ対応）|
|[latest_fig07b.pdf](data/latest_fig07b.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_07b.pdf)|最新ATLAS 12 SRのpost-fit score分布（§3.1にカテゴリ対応）|
|[latest_fig07c.pdf](data/latest_fig07c.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_07c.pdf)|最新ATLAS 12 SRのpost-fit score分布（§3.1にカテゴリ対応）|
|[latest_fig07d.pdf](data/latest_fig07d.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_07d.pdf)|最新ATLAS 12 SRのpost-fit score分布（§3.1にカテゴリ対応）|
|[latest_fig07e.pdf](data/latest_fig07e.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_07e.pdf)|最新ATLAS 12 SRのpost-fit score分布（§3.1にカテゴリ対応）|
|[latest_fig07f.pdf](data/latest_fig07f.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_07f.pdf)|最新ATLAS 12 SRのpost-fit score分布（§3.1にカテゴリ対応）|
|[latest_fig08a.pdf](data/latest_fig08a.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_08a.pdf)|最新ATLAS 12 SRのpost-fit score分布（§3.1にカテゴリ対応）|
|[latest_fig08b.pdf](data/latest_fig08b.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_08b.pdf)|最新ATLAS 12 SRのpost-fit score分布（§3.1にカテゴリ対応）|
|[latest_fig08c.pdf](data/latest_fig08c.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_08c.pdf)|最新ATLAS 12 SRのpost-fit score分布（§3.1にカテゴリ対応）|
|[latest_fig08d.pdf](data/latest_fig08d.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_08d.pdf)|最新ATLAS 12 SRのpost-fit score分布（§3.1にカテゴリ対応）|
|[latest_fig08e.pdf](data/latest_fig08e.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_08e.pdf)|最新ATLAS 12 SRのpost-fit score分布（§3.1にカテゴリ対応）|
|[latest_fig08f.pdf](data/latest_fig08f.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/fig_08f.pdf)|最新ATLAS 12 SRのpost-fit score分布（§3.1にカテゴリ対応）|
|[projection2024_full.pdf](data/projection2024_full.pdf)|[取得URL](https://inspirehep.net/files/735311e4e52d37c9117fc7cde5b69aff)|ATL-PHYS-PUB-2024-016全文、τ decay channel別投影|
|[latest_figaux20.pdf](data/latest_figaux20.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-2024-37/figaux_20.pdf)|Auxiliary Figure 20、Run2/3/combined期待／観測σ|
|[hepdata_legacy_record.yaml.tar.gz](data/hepdata_legacy_record.yaml.tar.gz)|[取得URL](https://www.hepdata.net/download/submission/ins2155171/1/yaml)|HEPData全submission archive（§4.1）|
|[atlas_2024_page.html](data/atlas_2024_page.html)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HDBS-2019-27/)|HDBS-2019-27入口のmeta-refresh HTML|
|[atlas_2024_index.html](data/atlas_2024_index.html)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HDBS-2019-27/index.php)|HDBS-2019-27の実ページ、別番号HEPDataへのリンク|
|[atlas_vbf_cp_2025.pdf](data/atlas_vbf_cp_2025.pdf)|[取得URL](https://arxiv.org/pdf/2506.19395)|ATLAS 2506.19395本文、HVV CP（τ Yukawaとは異なる）|
|[hepdata_projection_base_record.json](data/hepdata_projection_base_record.json)|[取得URL](https://www.hepdata.net/record/ins2779337?format=json)|HEPData数値／目録（§4.1）|
|[atlas_2024_tabaux01.pdf](data/atlas_2024_tabaux01.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HDBS-2019-27/tabaux_01.pdf)|投影元Auxiliary Table 01、ggF 6 BDTの全入力|
|[atlas_2024_tabaux02.pdf](data/atlas_2024_tabaux02.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HDBS-2019-27/tabaux_02.pdf)|投影元Auxiliary Table 02、VBF 3 BDTの全入力|
|[atlas_2024_tabaux03.pdf](data/atlas_2024_tabaux03.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HDBS-2019-27/tabaux_03.pdf)|投影元Auxiliary Table 03、SM/EFT A×ε|
|[hepdata_projection_base.yaml.tar.gz](data/hepdata_projection_base.yaml.tar.gz)|[取得URL](https://www.hepdata.net/download/submission/ins2779337/1/yaml)|HEPData全submission archive（§4.1）|
|[hepdata_projection_base_kl_expected.json](data/hepdata_projection_base_kl_expected.json)|[取得URL](https://www.hepdata.net/download/table/ins2779337/Combined%20%24-2%5Cln%5CLambda%24%20vs%20%24%5Ckappa_%7B%5Clambda%7D%24%20exp/json)|HEPData数値／目録（§4.1）|
|[inspire_atlas_polarimetry_thesis.json](data/inspire_atlas_polarimetry_thesis.json)|[取得URL](https://inspirehep.net/api/literature/1658537)|INSPIRE書誌／external identifiers／PDFリンク。検索結果の該当有無も保持|
|[tau_highpt_twospinner.pdf](data/tau_highpt_twospinner.pdf)|[取得URL](https://arxiv.org/pdf/1604.00964)|1604.00964本文、ττ＋jets polarization／EW scheme|
|[tau_ilc_chargedh.pdf](data/tau_ilc_chargedh.pdf)|[取得URL](https://arxiv.org/pdf/hep-ph/0507100)|hep-ph/0507100本文、ILC charged-H polarization spectrum|
|[tau_ilc_cp.pdf](data/tau_ilc_cp.pdf)|[取得URL](https://inspirehep.net/files/43ad945c04e57a1da5a340d59259721a)|1804.01241 publication PDF、ILD full simulation CP|
|[atlas_projection_base_fig6b.pdf](data/atlas_projection_base_fig6b.pdf)|[取得URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HDBS-2019-27/fig_06b.pdf)|2404.12660 Figure 6b、τhadτhad ggF Hi score|
|[atlas_polarimetry_thesis_2014.pdf](data/atlas_polarimetry_thesis_2014.pdf)|[取得URL](https://inspirehep.net/files/732c0f9e050fda120c785eadebb34c27)|BONN-IB-2014-03全文、reco H_Z＋H/Z BDT先例|
|[cms_latest_fig5.pdf](data/cms_latest_fig5.pdf)|[取得URL](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/HIG-25-008/CMS-PAS-HIG-25-008_Figure_005.pdf)|CMS Figure 5、Run-3カテゴリ別expected/observed upper limits|
|[cms_latest_fig6.pdf](data/cms_latest_fig6.pdf)|[取得URL](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/HIG-25-008/CMS-PAS-HIG-25-008_Figure_006.pdf)|CMS Figure 6、Run-3 κλ/κ2V依存cross-section limit curves|

### 4.3 失敗／確認画面の保存と未確定リンク

CERN CDSではCloudflareではなく**Anubis “Making sure you're not a bot!”**が返った。DESYはfast-challenge HTML。challengeを解く、cookieを偽装するなどの回避は行っていない。ATLAS noteの取得に使ったINSPIREは、そのrecord自身が公開している全文mirrorである。

|試行URL|結果|保存した応答／備考|
|---|---|---|
|[試行URL](https://cds.cern.ch/record/2925853/files/ATL-PHYS-PUB-2025-006.pdf)|HTTP 200、Anubis確認HTML。PDF取得失敗|[atlas_projection.response.html](data/atlas_projection.response.html)|
|[試行URL](https://cds.cern.ch/record/2928096/files/NOTE2025_006.pdf)|HTTP 200、Anubis確認HTML。PDF取得失敗|[cms_projection.response.html](data/cms_projection.response.html)|
|[試行URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PUBNOTES/ATL-PHYS-PUB-2025-006/ATL-PHYS-PUB-2025-006.pdf)|HTTP 404、HTML|[atlas_projection_direct.response.html](data/atlas_projection_direct.response.html)|
|[試行URL](https://cms-results.web.cern.ch/cms-results/public-results/publications/NOTE-2025-006/NOTE-2025-006.pdf)|HTTP 404、HTML|[cms_projection_direct.response.html](data/cms_projection_direct.response.html)|
|[試行URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGG-2018-40/)|HTTP 404、HTML|[atlas_legacy_page.html](data/atlas_legacy_page.html)。正しい公開コードHDBS-2018-40で取得成功|
|[試行URL](https://www.hepdata.net/record/ins3184980?format=json)|HTTP 404、HTML|[hepdata_latest_record.json](data/hepdata_latest_record.json)|
|[試行URL](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/HIG-25-008/HIG-25-008-pas.pdf)|HTTP 404、HTML|[cms_latest.response.html](data/cms_latest.response.html)|
|[試行URL](https://cms-results.web.cern.ch/cms-results/public-results/publications/NOTE-2025-006/)|HTTP 404、HTML|[cms_projection_page.html](data/cms_projection_page.html)|
|[試行URL](https://cds.cern.ch/record/2968133/files/HIG-25-008-pas.pdf)|HTTP 200、Anubis確認HTML。PDF取得失敗|[cms_latest_cds.response.html](data/cms_latest_cds.response.html)|
|[試行URL](https://cms-results.web.cern.ch/cms-results/public-results/publications/NOTE-2025-006/CMS-NOTE-2025-006.pdf)|HTTP 404、HTML|[cms_projection_public.response.html](data/cms_projection_public.response.html)|
|[試行URL](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/NOTE-2025-006/NOTE2025_006.pdf)|HTTP 404、HTML|[cms_projection_other.response.html](data/cms_projection_other.response.html)|
|[試行URL](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PUBNOTES/ATL-PHYS-PUB-2025-001/ATL-PHYS-PUB-2025-001.pdf)|HTTP 404、HTML|[projection_bbtautau.response.html](data/projection_bbtautau.response.html)。2025-001はbbγγ noteであり、bbττの正本ではない|
|[試行URL](https://bib-pubdb1.desy.de/record/640734/files/ATL-PHYS-PUB-2025-001.pdf)|HTTP 200、fast-challenge HTML。PDF取得失敗|[projection_bbtautau_desy.response.html](data/projection_bbtautau_desy.response.html)。2025-001はbbγγ noteであり、bbττの正本ではない|
|[試行URL](https://www.hepdata.net/record/ins2816777?format=json)|HTTP 404、HTML|[hepdata_2024_record.json](data/hepdata_2024_record.json)。ins2779337での取得は別途成功|
|[試行URL](https://cms-results.web.cern.ch/cms-results/public-results/plots/DP-2025-073/DP2025_073.pdf)|HTTP 404、HTML|[cms_tau_dp73.response.html](data/cms_tau_dp73.response.html)|
|[試行URL](https://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2024_241_v8.pdf)|HTTP 000、curl exit 60。詳細stderrは空|ファイル未保存。検索で得たdraft URLの試行であり、公開note正本との同一性は未確認|
|[試行URL](https://cds.cern.ch/record/2928096?ln=en)|HTTP 200、Anubis確認HTML。PDF取得失敗|[cms_note_cds_record.html](data/cms_note_cds_record.html)|
|[試行URL](https://cds.cern.ch/record/2946445/files/DP2025_073.pdf)|HTTP 200、Anubis確認HTML。PDF取得失敗|[cms_tau_dp73_cds.response.html](data/cms_tau_dp73_cds.response.html)|

[Bullock–Hagiwara–Martinのpublisher全文](https://www.sciencedirect.com/science/article/pii/055032139390045Q)はweb toolで403 Forbidden。本文ファイルは保存できず、INSPIREのpublisher abstractと書誌のみを使用した。CMS HIG-25-008のHEPData URLは未置換placeholderで、record番号を推測して収量を割り当てなかった。

文献検索はHH＋tau polarization、spin correlation、polarimetric vector、di-Higgs＋tau spinの組をarXiv/INSPIRE/CERN primary sitesで実施し、reference chainsも確認した。再現用の入口は[INSPIRE：Higgs pair tau polarization](https://inspirehep.net/literature?size=25&page=1&q=Higgs%20pair%20tau%20polarization)、[INSPIRE：di-Higgs polarimetric vector](https://inspirehep.net/literature?size=25&page=1&q=di-Higgs%20polarimetric%20vector)。これらの検索だけから不存在を断定せず、取得できた最も近い文献を§C/Dに列挙した。

## 5. 参考文献

本文を取得した文献には節・表・図を上記に記した。全文取得不能なnoteは、その事実を併記する。

|番号／arXiv・report number|原題|要点|
|---|---|---|
|[ATL-PHYS-PUB-2025-006](https://inspirehep.net/files/5cabc0e66dede649b8f6e37147e87a82)|Projected sensitivity of measurements of Higgs boson pair production with the ATLAS experiment at the HL-LHC|4.26σ、κλ 68% [0.58,1.48]の正本|
|[ATL-PHYS-PUB-2024-016](https://inspirehep.net/files/735311e4e52d37c9117fc7cde5b69aff)|Updated projection of the sensitivity of searches for Higgs boson pair production in the b b̄ τ⁺τ⁻ final state from LHC Run 2 to the High Luminosity LHC with the ATLAS detector|bbττのτ decay channel別・系統別投影|
|[2504.00672 / ATL-PHYS-PUB-2025-018 / CMS-HIG-25-002](https://arxiv.org/abs/2504.00672)|Highlights of the HL-LHC physics projections by ATLAS and CMS|CMS対応値と7.2/7.6σ joint projectionの仮定|
|[CMS-NOTE-2025-006](https://cds.cern.ch/record/2928096)|Projection of CMS experimental reach on HH production at HL-LHC|CMS正本。全文はAnubisで未取得|
|[2406.09971](https://arxiv.org/abs/2406.09971)|Combination of Searches for Higgs Boson Pair Production in pp Collisions at √s = 13 TeV with the ATLAS Detector|Run-2 HH組合せ。2404.12660とは異なる|
|[2404.12660](https://arxiv.org/abs/2404.12660)|Search for the non-resonant production of Higgs boson pairs via gluon fusion and vector-boson fusion in the b b̄ τ⁺τ⁻ final state in proton-proton collisions at √s = 13 TeV with the ATLAS detector|2025 HL-LHC bbττ投影の入力、HEPData ins2779337|
|[2607.26879](https://arxiv.org/abs/2607.26879)|Improved analysis of non-resonant Higgs boson pair production in the b b̄ τ⁺τ⁻ final state with 196 fb⁻¹ of data collected at √s = 13 TeV and 13.6 TeV with the ATLAS detector|最新ATLAS、Transformer＋Run2/3、12 SR|
|[2209.10910](https://arxiv.org/abs/2209.10910)|Search for resonant and non-resonant Higgs boson pair production in the b b̄ τ⁺τ⁻ decay channel using 13 TeV pp collision data from the ATLAS detector|legacyの44 score binsをHEPDataから取得|
|[CMS-PAS-HIG-25-008](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/HIG-25-008/)|Search for nonresonant HH production in the bbττ final state in proton-proton collisions at √s = 13.6 TeV|最新CMS、310 fb⁻¹ combined。arXivは公開ページのplaceholder|
|[2206.09401](https://arxiv.org/abs/2206.09401)|Search for nonresonant Higgs boson pair production in final state with two bottom quarks and two tau leptons in proton-proton collisions at √s = 13 TeV|CMS Run-2 DNN、DeepTau、limit 3.3(5.2)|
|[2608.10961](https://arxiv.org/abs/2608.10961)|TauPolaris: reconstructing tau lepton polarimetric vectors with conditional normalizing flows|最も近いh再構成、H/Z variables、CP指標18%改善|
|[BONN-IB-2014-03](https://inspirehep.net/literature/1658537)|Studies of Tau-Lepton Polarisation in Decays of Higgs and Z Bosons with the ATLAS Experiment|Roggendorfの修士論文。reco H_Z＋H/Z BDTの直接先例。arXivなし|
|[2110.04836](https://arxiv.org/abs/2110.04836)|Analysis of the CP structure of the Yukawa coupling between the Higgs boson and τ leptons in proton-proton collisions at √s = 13 TeV|CMS Run-2 IP/ρ/a₁ polarimetric-vector法|
|[2606.03510](https://arxiv.org/abs/2606.03510)|Analysis of the CP structure of the Yukawa coupling between the Higgs boson and tau leptons in proton-proton collisions at √s = 13.6 TeV|最新CMS CP、Run2+3 expected precision 14°|
|[2212.05833](https://arxiv.org/abs/2212.05833)|Measurement of the CP properties of Higgs boson interactions with τ-leptons with the ATLAS detector|ATLAS τ Yukawa CP、IPとdecay planes|
|[2506.19395](https://arxiv.org/abs/2506.19395)|Probing the Higgs boson CP properties in vector-boson fusion production in the H→τ⁺τ⁻ channel with the ATLAS detector|2025 ATLAS HVV CP。τ Yukawaの更新とは区別|
|[2201.08458](https://arxiv.org/abs/2201.08458)|Identification of hadronic tau lepton decays using a deep neural network|DeepTauのPF、IP/SV、47 high-level inputs|
|[CMS-DP-2025-073](https://cds.cern.ch/record/2946445)|Comparison of the performance of tau reconstruction and identification algorithms in Run 3|DeepTau/PNet/UParT比較。全文未取得、HIG採用は未確認|
|[Bullock–Hagiwara–Martin, NPB395 (1993) 499–533](https://doi.org/10.1016/0550-3213(93)90045-Q)|Tau polarization and its correlations as a probe of new physics|charged/neutral HとZのτ decay energy/correlation。arXivなし、全文未取得|
|[0808.0438](https://arxiv.org/abs/0808.0438)|Using Tau Polarization for Charged Higgs Boson and SUSY searches at LHC|charged-track fraction RによるW背景抑制|
|[1103.1827](https://arxiv.org/abs/1103.1827)|Improved Sensitivity to Charged Higgs Searches in Top Quark Decays t→bH⁺→b(τ⁺ντ) at the LHC using τ Polarisation and Multivariate Techniques|τ polarization＋BDTでH±/W識別|
|[1201.0117](https://arxiv.org/abs/1201.0117)|TauSpinner program for studies on spin effect in tau production at the LHC|W/Z/Hのlongitudinal polarization/correlation reweighting|
|[1406.1647](https://arxiv.org/abs/1406.1647)|TauSpinner: a tool for simulating CP effects in H→ττ decays at LHC|HとDYのtransverse spin modelling。abstractを参照|
|[1604.00964](https://arxiv.org/abs/1604.00964)|Production of tau lepton pairs with high pT jets at the LHC and the TauSpinner reweighting algorithm|Z polarizationのjets／EW scheme依存を数値で検証|
|[hep-ph/0507100](https://arxiv.org/abs/hep-ph/0507100)|Impact of tau polarization on the study of the MSSM charged Higgs bosons in top quark decays at the ILC|ILCのW/H±偏極差とπ spectrum fit|
|[1804.01241](https://arxiv.org/abs/1804.01241)|Measuring the CP state of tau lepton pairs from Higgs decay at the ILC|full simulationのpolarimeter再構成＋NN categorization＋CP likelihood|
