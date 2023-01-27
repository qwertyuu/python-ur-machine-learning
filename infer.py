from flask import Flask, jsonify, request
import joblib
from pandas import json_normalize

model_name = "model2.pkl"
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
    j = df['game'].str.replace('\s|\.', '', regex=True).str.replace('D', '3').str.replace('L', '1').str.replace('-', '2').str.split('', expand=True)
    j = j.drop([0, 21], axis=1).astype("int32") - 2
    k = ['game' + str(i) for i in range(20)]
    df[k] = j
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
