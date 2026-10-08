# 独立レビューと主解析への反映

GPT-6 mainが統合し、同providerの独立experimental-physicistとskeptical-reviewerが設計・実装・結果を読んだ。provider間独立ではない。最終文章の編集は結果確定後に独立Final Editorへ渡す。variant/effortはruntimeで未確認。

## 設計・実装

- 最新ATLASの独立d0 cut無し、SLTのみ、質量条件を一次文献から確認し、全armの選択を揃えた。初期のSLT/LTT試行は性能を読まずに中断・保存。
- 横track perigeeのd0とz0へ修正。nullは真の横d0のみを消し、縦IPと選択を本比較に一致させた。
- train-only scaling、object-type flags、validation-only停止とbin、Hi/Lo共有group nuisance、有限MC補助尤度を点検。一定scoreでも一つのbinを作る実装へ修正した。
- 各processと高score binのprompt率を出力し、flavor診断は両armのMC-support確認を伴うようにした。

## Spinの結果

主要4図をmainとvalidity reviewerが実画像で確認し、数値と表示を照合した。条件付きtoy比較として採用。KのAUC約0.51は全spin仮説でkinematicsを同じにしたためで、最新ATLAS baselineの再現ではない。Angular4はCP依存成分を捨てる。raw h normを校正済みconfidenceと呼ばず、seed0の固定model・binのMC区間を三seed平均へ付け替えない。

Skeptical reviewerの代替説明に基づき、平均hの外積を明示するProduct15 controlを追加。Joint15固有の上積みは95%条件付きMC区間で分離できず、「二体情報無し」「十分統計量」とは結論しない。mode flagsの有無を揃えるK+Modes controlでも+2.379%が残った。補助比較は結果後の探索で、主modelの採用変更ではない。両reviewerが定義・実装・数値とこのscopeを確認した。

## 寿命・最終文章

独立validity reviewerは寿命の全10 PNGとspinの全4 PNGを実画像で確認し、life表の点推定・68/95%区間、aligned bootstrapからのCombined/Had、Product15とK+ModesをJSONから再計算した。旧／修復後の24モデルのbin edges、validation／evaluation populationsは完全一致。本学習22 Transformerの点推定は収束・MC-supportを満たし、最終推論の2,296 profile fitは失敗ゼロ。GBDTとflavorのbaseline MC不足は参考診断のまま保持した。

尤度修復については、独立65桁のobjective／gradient検証、synthetic144条件、解析Hessian、全24templateのtheta+logbeta同時fit、統計のみの閉形式、二つの別初期点を点検した。全24templateで独立fitとのZ相対差は最大1.88×10⁻¹⁴、統計のみとは1.20×10⁻¹⁴。元の数値失敗を物理結果へ読み替えず、モデル・score・bin・nuisanceを固定した再計算として採用した。

最終validity verdictは限定された探索的toyの範囲でapprove-with-conditions。標準electron likelihoodの文献にd0変数があることから、最新HH campaignで有効な設定まで確認済みと書かない条件を、報告とprotocolへ反映した。実際のID・isolationの変位依存は未検証と明示した。bin図の凡例が一部の上側barと重なる点は記録する。高scoreのsignal／MC比較は読めるため探索報告を妨げないが、publication／presentation exportでは凡例を枠外へ移す。

最終skeptical reviewerは報告・protocol・最終JSON・実際の6 PNGを独立に確認し、approve-with-conditions。非即発stressの比0.873はseed 0・固定learnerの点推定でpaired区間がないため、一般的な寿命情報の消失と読ませない条件を反映した。同条件のstat-only比は約1.005。Product15との未分離は固定learner／seed 0の結果に限定する。有限MC・初期値・背景・未検証IDの制約を併記する現在の結論には、追加の重い実験は不要との判定。

## 最終編集と主解析の照合

独立Final Editorは完成稿を一度読み、大きな組替えは不要と判定した。冒頭を二段落に分け、Z比・二体積・baseline B・Hi/Loの定義を数値より先に置き、寿命節に三つの小見出しを加えた。幾何closureの精密な数値を補足へ移し、最終判定の用語を本文に揃えた。原稿にない因果・事実は加えていない。

mainは編集後の表、bootstrap区間、nonpromptのstat-only比、spin controls、図の説明と適用範囲を最終JSONへ戻して照合した。旧ATLAS換算を現在値として残さず、標準LH定義と最新campaign設定を区別し、nonprompt単一seedとProduct15未分離の条件を保った。Projectへの還元はこの照合後に行った。
