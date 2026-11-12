.PHONY: all run report slides viva clean

PYTHON ?= python3

all: run report slides viva

run:
	$(PYTHON) run_pipeline.py

report:
	cd report && TEXMFVAR=/tmp/texmf-var pdflatex -interaction=nonstopmode technical_report.tex && TEXMFVAR=/tmp/texmf-var pdflatex -interaction=nonstopmode technical_report.tex

slides:
	cd report && TEXMFVAR=/tmp/texmf-var pdflatex -interaction=nonstopmode Presentation_Slides.tex && TEXMFVAR=/tmp/texmf-var pdflatex -interaction=nonstopmode Presentation_Slides.tex

viva:
	cd report && TEXMFVAR=/tmp/texmf-var pdflatex -interaction=nonstopmode Viva_Guide_and_Explanation_Sheet.tex && TEXMFVAR=/tmp/texmf-var pdflatex -interaction=nonstopmode Viva_Guide_and_Explanation_Sheet.tex

clean:
	rm -rf src/__pycache__ report/*.aux report/*.log report/*.out
