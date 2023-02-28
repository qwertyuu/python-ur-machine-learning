from flask import Flask, jsonify, request
import pickle
from pandas import json_normalize
from src.prep import convert_game_to_cols
import os
import requests
import shutil
import dotenv

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
    with open(model_name, 'rb') as f:
        model = pickle.load(f)
    return model

model_pool = [
    load(),
]

@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json()
    df = json_normalize(data)

    df = convert_game_to_cols(df)
    
    df = df.drop(columns=["game"])
    if len(model_pool) == 0:
        model = load()
    else:
        model = model_pool.pop()
    q = model.predict(df)
    model_pool.append(model)
    return jsonify(utilities=list(q))

if __name__ == '__main__':
    app.run("0.0.0.0", 5000)
