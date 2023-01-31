from flask import Flask, jsonify, request
import joblib
from pandas import json_normalize
from src.prep import convert_game_to_cols

model_name = "model_rank.pkl"
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
    if len(model_pool) == 0:
        print("creating model")
        model = joblib.load(model_name)
    else:
        model = model_pool.pop()
    game_col = df.pop("game")
    predictions = model.predict(df, group=[len(df)])
    df["pred"] = predictions
    df["game"] = game_col
    df.sort_values("pred", ascending=False, inplace=True)
    orig_test_data_grouped = df.groupby(["game", "roll", "light_score", "dark_score", "light_left", "dark_left", "light_turn"])
    assert orig_test_data_grouped.ngroups == 1
    print(df)

    model_pool.append(model)
    return jsonify(utilities=df[["x", "y", "pred"]].to_dict('records'))

if __name__ == '__main__':
    app.run()
