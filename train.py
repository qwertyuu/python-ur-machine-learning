import pandas as pd
import lightgbm as lgb
import joblib


def run():
    data = pd.read_parquet("data/dataset_depth8_Sam_Raph_Sothatsit3.parquet")
    data.pop("game")
    data.pop("rank")
    label = "utility"
    #game_fields = ["game0","game1","game2","game3","game4","game5","game6","game7","game8","game9","game10","game11","game12","game13","game14","game15","game16","game17","game18","game19"]
    #for game_field in game_fields:
    #    data[game_field] = data[game_field] + 1
#
    #cat_fields = [
    #    *game_fields,
    #    "light_turn"
    #]
    train_data = data.sample(frac=0.8, random_state=1)
    test_data = data.drop(train_data.index)
    # Convert the data into LightGBM's format
    train_dataset = lgb.Dataset(train_data.drop(
        label, axis=1), label=train_data[label])
    test_dataset = lgb.Dataset(test_data.drop(
        label, axis=1), label=test_data[label])

    # Set the model parameters
    params = {
        #"num_iterations": 100000,
        "metric": "rmse",
        "objective": "regression",
        "early_stopping_round": 15,
        'feature_fraction': 0.8, 
        'learning_rate': 0.01,
        'num_leaves': 1000,
        'max_bin': 10,
        'min_child_samples': 10,
        #'boosting': 'dart',
    }

    # Train the model
    model = lgb.train(params, train_dataset, valid_sets=[test_dataset], num_boost_round=10000)
    # save model
    joblib.dump(model, 'lgb_sleep.pkl')

    # Make predictions on the test set
    predictions = model.predict(test_data.drop(label, axis=1))
    print(pd.DataFrame({"pred": predictions, "test": test_data[label]}))


if __name__ == "__main__":
    run()
