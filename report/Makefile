LATEXMK ?= latexmk
CACHE_DIR ?= .tex-cache
LATEXMK_FLAGS := -xelatex -interaction=nonstopmode -halt-on-error

.PHONY: all docs outline analysis review draft clean

all: main.pdf

docs: main.pdf outline.pdf analysis_note.pdf physics_review.pdf draft_v1.pdf

main.pdf: main.tex preamble.tex references.bib
	mkdir -p "$(CACHE_DIR)"
	XDG_CACHE_HOME="$(abspath $(CACHE_DIR))" $(LATEXMK) $(LATEXMK_FLAGS) main.tex

outline: outline.pdf

outline.pdf: outline.tex preamble.tex
	mkdir -p "$(CACHE_DIR)"
	XDG_CACHE_HOME="$(abspath $(CACHE_DIR))" $(LATEXMK) $(LATEXMK_FLAGS) outline.tex

analysis: analysis_note.pdf

analysis_note.pdf: analysis_note.tex preamble.tex
	mkdir -p "$(CACHE_DIR)"
	XDG_CACHE_HOME="$(abspath $(CACHE_DIR))" $(LATEXMK) $(LATEXMK_FLAGS) analysis_note.tex

review: physics_review.pdf

physics_review.pdf: physics_review.tex preamble.tex
	mkdir -p "$(CACHE_DIR)"
	XDG_CACHE_HOME="$(abspath $(CACHE_DIR))" $(LATEXMK) $(LATEXMK_FLAGS) physics_review.tex

draft: draft_v1.pdf

draft_v1.pdf: draft_v1.tex preamble.tex references.bib
	mkdir -p "$(CACHE_DIR)"
	XDG_CACHE_HOME="$(abspath $(CACHE_DIR))" $(LATEXMK) $(LATEXMK_FLAGS) draft_v1.tex

clean:
	$(LATEXMK) -C main.tex
	$(LATEXMK) -C outline.tex
	$(LATEXMK) -C analysis_note.tex
	$(LATEXMK) -C physics_review.tex
	$(LATEXMK) -C draft_v1.tex
