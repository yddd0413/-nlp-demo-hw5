@echo off
echo ========================================
echo   NLP Advanced Tasks - Setup and Run
echo ========================================
echo.

echo [Step 1/3] Installing Python packages...
pip install streamlit requests spacy torch transformers

echo.
echo [Step 2/3] Downloading spaCy model...
python -m spacy download en_core_web_sm

echo.
echo [Step 3/3] Starting Streamlit app...
echo.
echo App will open in your browser at: http://localhost:8501
echo Press Ctrl+C to stop the server
echo ========================================
echo.

streamlit run nlp_app.py
