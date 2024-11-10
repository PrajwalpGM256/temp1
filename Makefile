# Makefile

# Variables
DATA_DIR = data
DATA_FILE = diamonds.csv
DATA_URL = https://raw.githubusercontent.com/tidyverse/ggplot2/main/data-raw/diamonds.csv

# Phony targets
.PHONY: all data quest1 quest2 quest3 quest4 quest5

# Default target
all: data quest1 quest2 quest3 quest4 quest5

# Rule to download the data
data: $(DATA_DIR)/$(DATA_FILE)

$(DATA_DIR)/$(DATA_FILE):
	@mkdir -p $(DATA_DIR)
	@echo "Downloading data..."
	@wget -q -O $(DATA_DIR)/$(DATA_FILE) $(DATA_URL)
	@echo "Data downloaded to $(DATA_DIR)/$(DATA_FILE)"

# Quest targets
quest1:
	python -B src/quest1.py

quest2:
	python -B src/quest2.py

quest3:
	python -B src/quest3.py

quest4:
	python -B src/quest4.py

quest5:
	python -B src/quest5.py
