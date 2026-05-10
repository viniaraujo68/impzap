PY ?= python3
SRC := src
OUT := out

SCRIPTS := \
	$(SRC)/slide_05_escada_pontos.py \
	$(SRC)/slide_06_arquitetura.py \
	$(SRC)/slide_08_matriz_paradigmas.py \
	$(SRC)/slide_09_reinforce_curve.py \
	$(SRC)/slide_12_jornada_cfr.py \
	$(SRC)/slide_13_cfr_v9_oponentes.py \
	$(SRC)/slide_16_despacho_hmm_cfr.py

PNGS := $(patsubst $(SRC)/%.py,$(OUT)/%.png,$(SCRIPTS))

.PHONY: all clean help

all: $(PNGS)

$(OUT)/%.png: $(SRC)/%.py $(SRC)/_style.py
	@mkdir -p $(OUT)
	$(PY) $<

clean:
	rm -rf $(OUT)

help:
	@echo "Targets:"
	@echo "  all   - regenerate every slide PNG"
	@echo "  clean - remove out/ directory"
	@echo "  help  - this message"
	@echo ""
	@echo "Per-slide: make out/slide_12_jornada_cfr.png"
