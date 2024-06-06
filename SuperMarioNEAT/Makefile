# Define the directories
MODELS_DIR := models/
CHECKPOINTS_DIR := checkpoints/
REPORTS_DIR := reports/

# Default target
.PHONY: help
help:
	@echo "Usage: make [TARGET]"
	@echo "Targets:"
	@echo "  clean_models       - Remove the models directory"
	@echo "  clean_checkpoints  - Remove the checkpoints directory"
	@echo "  clean_reports      - Remove the reports directory"
	@echo "  clean_all          - Remove all directories"

# Target to clean models directory
.PHONY: clean_models
clean_models:
	@if [ -d "$(MODELS_DIR)" ]; then \
		echo "Removing $(MODELS_DIR)..."; \
		rm -rf $(MODELS_DIR); \
		echo "$(MODELS_DIR) removed."; \
	else \
		echo "$(MODELS_DIR) does not exist."; \
	fi

# Target to clean checkpoints directory
.PHONY: clean_checkpoints
clean_checkpoints:
	@if [ -d "$(CHECKPOINTS_DIR)" ]; then \
		echo "Removing $(CHECKPOINTS_DIR)..."; \
		rm -rf $(CHECKPOINTS_DIR); \
		echo "$(CHECKPOINTS_DIR) removed."; \
	else \
		echo "$(CHECKPOINTS_DIR) does not exist."; \
	fi

# # Target to clean reports directory
# .PHONY: clean_reports
# clean_reports:
# 	@if [ -d "$(REPORTS_DIR)" ]; then \
# 		echo "Removing $(REPORTS_DIR)..."; \
# 		rm -rf $(REPORTS_DIR); \
# 		echo "$(REPORTS_DIR) removed."; \
# 	else \
# 		echo "$(REPORTS_DIR) does not exist."; \
# 	fi

# Target to clean log file
.PHONY: clean_log
clean_log:
	@if [ -f "log.txt" ]; then \
		echo "Removing log.txt..."; \
		rm log.txt; \
		echo "log.txt removed."; \
	else \
		echo "log.txt does not exist."; \
	fi

# Target to clean all directories
.PHONY: clean_all
clean_all: clean_models clean_checkpoints clean_log