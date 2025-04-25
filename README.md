# Universal-Classification-Studio
🧠 Universal Classification Studio
A single-file Streamlit application that turns any tabular CSV into a live, downloadable probability model in seconds.

What it does
Upload a CSV with numeric and/or categorical columns.

Pick the target (binary or multi-class).

Select an algorithm:

Logistic Regression

Gradient Boosting

Random Forest

Train – the app handles preprocessing (impute ▸ scale ▸ one-hot encode) and splits data into train / test.

See accuracy + ROC-AUC, an interactive ROC curve (binary), and a table of scored rows.

Download the scored CSV and the champion model (model.pkl) for later use.

POC ONLY – no hyper-parameter tuning, bias checks, or governance.
Need production-grade MLOps? → drtomharty.com/bio

Features at a glance

Feature	Value
One-click model	Pipeline + Train test split
Mixed data types	Numeric imputer + scaler, categorical imputer + OHE
Three classifiers	LogReg, GBM, RandomForest
Metrics	Accuracy, ROC-AUC, ROC curve
Outputs	scored_data.csv + model.pkl
Stateless demo	Everything in RAM; browser tab session only
Quick start (local)
bash
Copy
Edit
git clone https://github.com/THartyMBA/universal-classification-studio.git
cd universal-classification-studio
python -m venv venv && source venv/bin/activate   # Win: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
Visit http://localhost:8501 → upload a CSV → experiment!

Deploy free on Streamlit Cloud
Push this folder to GitHub.

Go to streamlit.io/cloud → New app → pick repo / branch → Deploy.

(Optional) Add secrets if you extend the app with API keys.

Requirements
nginx
Copy
Edit
streamlit
pandas
numpy
scikit-learn
plotly
(Lightweight – runs on Streamlit Cloud’s free CPU)

Repo layout
bash
Copy
Edit
/app.py            ← the whole app
/requirements.txt
/README.md
License
CC0 – public-domain dedication. Attribution appreciated but not required.

Acknowledgements
Streamlit for the effortless UI

scikit-learn powering the models

Plotly for interactive visuals

Happy modeling! 🚀
