# 等質量 H/Z → ττ 分離レポート

ICEPPサマースクール実習の最終結果を、日本語のHEP論文形式でまとめた
TeX文書です。

## ファイル

- `main.tex` / `main.pdf`: レビューを反映した最終稿
- `outline.tex` / `outline.pdf`: 章立て、各章の役割、図表計画
- `draft_v1.tex` / `draft_v1.pdf`: 物理レビュー前の保存草稿
- `analysis_note.tex` / `analysis_note.pdf`: 根拠、仮定、レビュー反映記録
- `physics_review.tex` / `physics_review.pdf`: 独立物理・論文形式レビューの記録
- `references.bib`: 参考文献
- `figures/`: 解析repositoryから採用したPDF図

## ビルド

XeLaTeX、`bxjsarticle`、`latexmk`、upBibTeXを使います。

```sh
make
make docs
```

TeX Live 2026で `main.pdf` と `outline.pdf` のビルドを確認しています。
iceppサーバーには作成時点でTeX環境がないため、PDFは別環境で組版しました。

## 結果の範囲

本文が報告するのは、質量を約91.2 GeVに揃え、各分割・truth親ボソン
横運動量20 GeV binのH/Z事象数を等しくした専用Monte Carlo標本に対する、
再構成候補レベルの探索的分類性能です。スピン相関だけの分離能、
blind holdout性能、実データ感度、系統不確かさを測定したものではありません。
