from flask import Flask, jsonify, request
import pickle
import lightgbm as lgb
from pandas import json_normalize
from src.prep import convert_game_to_cols

model_name = 'lgb_classifier.txt'
app = Flask(__name__)

def load():
    return lgb.Booster(model_file=model_name)

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
        print("creating model")
        model = load()
    else:
        model = model_pool.pop()
    q = model.predict(df)
    model_pool.append(model)
    return jsonify(utilities=list(q))

if __name__ == '__main__':
    app.run()
