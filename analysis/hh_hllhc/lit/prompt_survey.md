あなたは素粒子実験（ATLAS、Higgs対生成、τ物理）の文献調査担当です。作業は読み取りと調査だけで、このディレクトリに報告ファイルを書く以外の変更はしないでください。日本語で書いてください（物理量・固有名・表番号は原文のまま）。

## 背景

京都大学ATLASの学生が「tauspin」という研究をしています。ATLAS full simulationのH→ττ（τhad）で、τのpolarimetric vector h（各τの崩壊からspinを測るベクトル）を、可視粒子・MET・PV基準のtrack impact parameter（IP）・3-prongのsecondary vertex（SV）から回帰する手法です。到達点は、等質量（125 GeV）に揃えたH→ττとZ→ττの分離で固定readout AUC 0.626（exact hで0.72〜0.74）。

これをHL-LHCのHH→bbττ探索に応用して、SM HH期待有意度（ATLAS単独で約4.3σと聞いている）をどれだけ上げられるかを、**最新の実解析を基準に、尤度レベルで**定量化したい。過去の簡易見積もりでは「最感度binの背景の組成（single H、Z+HF、tt̄、fake）が分からない」「HL-LHC投影の元解析と組成の元解析が違う」ために、換算がシナリオ幅に留まった。今回はそれを公開情報で埋め切るのが目的です。「調べる必要があります」で終わらせず、一次資料を開いて数値・表番号・図番号・URLまで確定させてください。確定できないものは、何を試してどこで止まったか（URL、エラー）を書いてください。

## 調べること（優先順）

### A. HL-LHC HH投影の正本
1. ATLASの最新HL-LHC HH投影（SM HH期待有意度、κλ区間）。「4.3σ」はどの文書のどの数字か（ATL-PHYS-PUB-番号、European Strategy 2025/2026 input、arXiv:2504.xxxx など）。bbbb・bbττ・bbγγ・その他の各チャンネル寄与、bbττのτhτh／τlτh別の値、系統誤差シナリオ（baseline、S1/S2、「stat only」など）ごとの数字。ATLAS+CMS組合せの数字。
2. その投影がどの実解析（Run-2 legacy bbττ arXiv:2209.10910、2404.12660 組合せ、Run2+3 arXiv:2607.26879 など）を元に、どういう外挿（ルミノシティ、√s 14 TeV、系統の縮小、ITk、b-tag/τ ID改善）で作られたか。外挿の具体的規則（どの系統をどれだけ縮めたか、MC統計の扱い）。
3. CMSの対応する最新投影（bbττ単独とHH組合せ）。

### B. 最新HH→bbττ実解析の詳細（ATLAS arXiv:2607.26879 を中心に、CMS最新bbττも）
1. 事象選択、カテゴリ（τhτh、τlτh SLT/LTT、m_HHのHi/Lo、VBFなど）、MVA（Transformer等）の入力変数一覧と、spin・polarisation・IP関連の入力の有無。
2. **最終fitのscore binごとの、process別（HH、single H（ggF/VBF/ZH/ttH）、tt̄（真τ／fake τ）、Z+HF、fake τ（multijet、tt̄ fake）、その他）の収量**。HEPDataに「score分布のprocess別収量」「pyhf/HistFactory JSON の尤度」があるか。あるならrecord番号・table名・取得URL（`https://www.hepdata.net/record/insXXXXXXX?format=json` や `https://www.hepdata.net/download/table/...`）。補助資料（tabaux_XX.pdf、figaux）にbin別組成があるか。2209.10910（Run-2 legacy）についても同様（Table 5 以外にbin別のprocess収量・HEPDataのpyhf尤度）。
3. 各チャンネル・カテゴリの期待有意度（観測でなく期待）と、最感度binの S/B。
4. 主要系統誤差とその影響の大きさ（ranking）、背景正規化をfreeにしているか（tt̄、Z+HF）。

### C. τ spin・polarisationをHH・H→ττ・背景除去に使う先行研究
1. HH→bbττでτ偏極・spin相関を背景除去に使う提案（現象論論文、実験note）。tt̄背景のτ偏極（W由来でP=−1）、Z→ττの偏極（P≈−0.15）、H→ττの横spin相関を使う提案。
2. TauPolaris（arXiv:2608.10961）がH/Z分離・背景抑制について何を示したか（数値）。
3. ATLAS/CMSのH→ττ CP解析（φ*_CP、IP法、ρ法、polarimetric vector法）の最新版と、そこで実際に使われたpolarimeter再構成の性能（CMS arXiv:2110.04836 等、ATLAS 2022/2025 CP）。
4. CMSのHH→bbττ（Run-2 2206.09401、最新Run2+3があれば）とCMSのτ ID（DeepTau, ParT）で、spinに相当する情報を使っているか。
5. 「τ偏極をZ/γ*→ττ背景の抑制、tt̄抑制に使う」古典的提案（Bullock-Hagiwara-Martin 1993、Hagiwara 系、ILC/LHC τ polarization at high pT 等）の要点。

### D. 本研究の新規性判定に必要な最近接先行研究
上のCの中から、「再構成水準のτ polarimetric vectorをHH→bbττの多変量分類に入れてHL-LHC感度を出した」研究が存在するか。あれば何が違うか、なければ最も近いものは何か。

## 出力

`/Users/ryunosuke/Projects/tauspin-wt-hllhc/analysis/hh_hllhc/lit/survey_report.md` に書く。構成：
1. 要約（10行以内）：HL-LHC投影の正本数値、元解析、bin別組成の取得可否、新規性の判定。
2. A〜D の各項目について、確定事実（一次資料URL＋節・表・図番号＋数値）と、確定できなかった事項（試したこと）を分けて書く。
3. 表：最新ATLAS bbττ と Run-2 legacy の、カテゴリ×score bin（取れる限り）のprocess別収量またはその取得先。
4. 機械取得できるデータ（HEPData JSON/YAML、pyhf JSON、補助表PDF）のURL一覧。実際にダウンロードできたものは `/Users/ryunosuke/Projects/tauspin-wt-hllhc/analysis/hh_hllhc/lit/data/` に保存し、ファイル名と中身の概要を書く。Cloudflareなどの確認画面は回避しない（回避できなかったと書く）。HEPDataはまず `curl -sL -H "Accept: application/json" "https://www.hepdata.net/record/insXXXXXXX?format=json"` や `https://www.hepdata.net/search/?q=...&format=json` を試すこと。arXivのHTML（`https://arxiv.org/html/XXXX`）やCDSのPDF、ATLAS公開ページ（`https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/HIGP-.../`）の補助表は直接取得してよい。
5. 参考文献リスト（arXiv番号、タイトル、要点1行）。

推測は推測と明記し、数値は必ず出典の該当箇所と一緒に書いてください。
