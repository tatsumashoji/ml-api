from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr


app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False
)

# モデル読み込み（起動時に一度だけ）
with open('model_m_PCEpiclockAge.pkl', 'rb') as f:
    model_m = pickle.load(f)

# モデル読み込み（起動時に一度だけ）
with open('model_f_PCEpiclockAge.pkl', 'rb') as f:
    model_f = pickle.load( f)

@app.route("/")
def health():
    return "OK"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # 例: data["answers"] が 52項目のリスト
    answers = data["answers"]  # [a1, a2, ..., a52]

    # 必要に応じて前処理
    # X = preprocess(answers)
    sex = answers[0]
    age = answers[1]
    X = [answers[2:] + [age]]

    if int(sex) == 0:
        y_pred = model_m.predict(X)[0]
    else:
        y_pred = model_f.predict(X)[0]

    return jsonify({"prediction": f'{round(float(y_pred), 2)}歳'})
