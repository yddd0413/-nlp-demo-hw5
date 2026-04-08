# NLP Advanced Tasks Demo

## Features

### Tab 1: EDU Segmentation
- Fetch data from NeuralEDUSeg GitHub repository
- Rule-based segmentation using spaCy
- Visual comparison between baseline and ground truth

### Tab 2: Shallow Discourse Parsing
- Identify explicit discourse markers
- Classify markers into PDTB categories
- Extract arguments (Arg1 and Arg2)

### Tab 3: Coreference Resolution
- Use fastcoref for entity clustering
- Highlight coreference chains
- Display equivalent classes

## Installation

### Method 1: Automatic Installation
```bash
run.bat
```

### Method 2: Manual Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run nlp_app.py
```

## Requirements
- Python 3.8+
- 8GB+ RAM recommended
- Internet connection for model downloads
