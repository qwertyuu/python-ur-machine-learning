import pandas as pd
import lightgbm as lgb
import joblib


def run():
    data = pd.read_parquet("data/dataset3.parquet")
    data.pop("game")
    data.pop("rank")
    label = "utility"
    train_data = data.sample(frac=0.8, random_state=1)
    test_data = data.drop(train_data.index)
    # Convert the data into LightGBM's format
    train_dataset = lgb.Dataset(train_data.drop(
        label, axis=1), label=train_data[label])
    test_dataset = lgb.Dataset(test_data.drop(
        label, axis=1), label=test_data[label])

    # Set the model parameters
    params = {"num_iterations": 10000, "num_leaves": 300, "learning_rate": 0.2, "metric": "rmse", "objective": "regression", "early_stopping_round": 15}

    # Train the model
    model = lgb.train(params, train_dataset, valid_sets=[test_dataset])
    # save model
    joblib.dump(model, 'lgb.pkl')

    # Make predictions on the test set
    predictions = model.predict(test_data.drop(label, axis=1))
    print(pd.DataFrame({"pred": predictions, "test": test_data[label]}))


if __name__ == "__main__":
    run()
