import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import GroupShuffleSplit


def run():
    data = pd.read_parquet("data/dataset3.parquet")
    label = "rank"
    group_keys = ["game", "roll", "light_score", "dark_score", "light_left", "dark_left", "light_turn"]
    g_data = data.groupby(group_keys)
    g_data.nth(1)
    splitter = GroupShuffleSplit(test_size=.20, n_splits=1, random_state = 7)
    split = splitter.split(data, groups=data[["game", "roll", "light_score", "dark_score", "light_left", "dark_left", "light_turn"]].astype(str).agg('-'.join, axis=1))
    train_inds, test_inds = next(split)

    train_data = data.iloc[train_inds]
    test_data = data.iloc[test_inds]

    train_grouped = train_data.groupby(group_keys)
    test_grouped = test_data.groupby(group_keys)
    train_group = train_grouped.size()
    test_group = test_grouped.size()
    train_data.pop("game")
    test_data.pop("game")
    train_data.pop("utility")
    test_data.pop("utility")

    # Convert the data into LightGBM's format
    train_dataset = lgb.Dataset(train_data.drop(
        label, axis=1), label=train_data[label], group=train_group)
    test_dataset = lgb.Dataset(test_data.drop(
        label, axis=1), label=test_data[label], group=test_group)

    # Set the model parameters
    params = {
        "num_iterations": 1000,
        "num_leaves": 100,
        "learning_rate": 0.3,
        'objective': 'rank_xendcg',
        'metric': 'ndcg',
        "early_stopping_round": 15, 
        "eval_at": "5,30,60"
    }

    # Train the model
    model = lgb.train(params, train_dataset, valid_sets=[test_dataset])
    # save model
    joblib.dump(model, 'lgb.pkl')

    orig_test_data = data.loc[test_data.index]

    # Make predictions on the test set
    predictions = model.predict(test_data.drop(label, axis=1))
    orig_test_data["pred"] = predictions
    orig_test_data.sort_values("pred", ascending=False, inplace=True)
    orig_test_data_grouped = orig_test_data.groupby(group_keys)
    print(orig_test_data_grouped.nth(1)[label].value_counts())


if __name__ == "__main__":
    run()
