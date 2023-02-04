from flask import Flask, jsonify, request
import joblib
from pandas import json_normalize
from src.prep import convert_game_to_cols

model_name = "model_8depth_2.1mil_0.58rmse.pkl"
app = Flask(__name__)
model_pool = [
    joblib.load(model_name),
    #joblib.load(model_name),
    #joblib.load(model_name),
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
    # game_fields = ["game0","game1","game2","game3","game4","game5","game6","game7","game8","game9","game10","game11","game12","game13","game14","game15","game16","game17","game18","game19"]
    # for game_field in game_fields:
    #     df[game_field] = df[game_field] + 1
    q = model.predict(df)
    model_pool.append(model)
    return jsonify(utilities=list(q))

if __name__ == '__main__':
    app.run()
