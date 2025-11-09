Flask Frontend App
==================

Files:
- app.py             -> Flask application
- templates/         -> HTML templates (index.html, result.html)
- static/style.css   -> Simple CSS
- model.joblib       -> Optional scikit-learn model (if present predictions work)
- requirements.txt   -> Python dependencies

Instructions:
1. Unzip or extract the folder.
2. (Optional) Put your scikit-learn model saved with joblib as 'model.joblib' into the folder.
3. Create a virtual environment:
   python -m venv venv
   source venv/bin/activate   (Linux/macOS)
   venv\Scripts\activate    (Windows)
4. Install dependencies:
   pip install -r requirements.txt
5. Run:
   python app.py
6. Open http://127.0.0.1:5000 in your browser.