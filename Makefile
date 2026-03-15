# ==============================================================================
# ImpZap - Makefile
# ==============================================================================

ENGINE_DIR = engine
ENV_DIR = truco_env
LIB_NAME = trucolib.so
HEADER_NAME = trucolib.h
PYTHON = python3

VENV_DIR = .venv

.PHONY: all build clean clean-venv clean-all setup

all: build

build:
	@echo "Compiling the Go engine with CGO..."
	@cd $(ENGINE_DIR) && go build -buildmode=c-shared -o ../$(ENV_DIR)/$(LIB_NAME) .
	@echo "Compilation finished! Binary located at $(ENV_DIR)/$(LIB_NAME)"

clean:
	@echo "Removing compiled artifacts..."
	@rm -f $(ENV_DIR)/$(LIB_NAME) $(ENV_DIR)/$(HEADER_NAME)
	@echo "Cleanup completed."

clean-venv:
	@echo "Deleting virtual environment..."
	@rm -rf $(VENV_DIR)
	@echo "Virtual environment removed."

clean-all: clean clean-venv
	@$(PYTHON) -m venv $(VENV_DIR)
	@echo "Deep clean completed. A fresh .venv has been created."
	@echo "Remember to run 'source .venv/bin/activate' and then 'make setup'!"

setup:
	@echo "Installing dependencies into the virtual environment..."
	@pip install -e .
	@echo "Environment setup completed."