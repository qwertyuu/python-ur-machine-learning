from flask import Flask, jsonify, request
import joblib
from pandas import json_normalize
from src.prep import convert_game_to_cols

model_name = "model4.pkl"
app = Flask(__name__)
model_pool = [
    joblib.load(model_name),
    joblib.load(model_name),
    joblib.load(model_name),
]


@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json()
    df = json_normalize(data)

    df = convert_game_to_cols(df)
    
    df = df.drop(columns=["game"])
    if len(model_pool) == 0:
        print("creating model")
        model = joblib.load(model_name)
    else:
        model = model_pool.pop()
    q = model.predict(df)
    model_pool.append(model)
    return jsonify(utilities=list(q))

if __name__ == '__main__':
    app.run()
