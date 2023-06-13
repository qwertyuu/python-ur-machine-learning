import time
from flask import Flask, jsonify, request
import pickle
import flask
import joblib
from pandas import json_normalize
from src.prep import convert_game_to_cols
import os
import requests
import shutil
import dotenv
import lightgbm as lgb

dotenv.load_dotenv()

model_name = os.getenv("MODEL_NAME", "model.pkl")
app = Flask(__name__)

# check that model name is local file or url. If url, download it to local file and use that
if model_name.startswith("http"):
    r = requests.get(model_name, stream=True)
    if r.status_code == 200:
        print("Downloading model from url")
        with open("model.pkl", 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
        model_name = "model.pkl"
    else:
        raise Exception("Could not download model from url")
    
def load():
    print("loading model")
    model = None
    if model_name.endswith(".pkl"):
        with open(model_name, 'rb') as f:
            model = pickle.load(f)
    elif model_name.endswith(".joblib"):
        model = joblib.load(model_name)
    elif model_name.endswith(".txt"):
        model = lgb.Booster(model_file=model_name)
    else:
        raise Exception("Model file must be .pkl, .joblib, or .txt")
    return model

max_model_pool_size = 1
d = {
    "current_model_pool_size": 0,
}
model_pool = [
    load(),
]

@app.route('/infer', methods=['POST'])
def infer():
    global max_model_pool_size, d
    data = request.get_json()
    df = json_normalize(data)

    df = convert_game_to_cols(df)
    
    df = df.drop(columns=["game"])
    while True:
        if len(model_pool) == 0:
            if d["current_model_pool_size"] >= max_model_pool_size:
                time.sleep(1)
                print("Waiting")
                continue
            d["current_model_pool_size"] += 1
            model = load()
            break
        else:
            model = model_pool.pop()
            break
    q = model.predict(df)
    model_pool.append(model)
    return jsonify(utilities=list(q))

if __name__ == '__main__':
    app.run("0.0.0.0", 5000)
