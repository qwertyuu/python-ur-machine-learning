import pandas as pd
import lightgbm as lgb
import joblib


def run():
    print("Loading data...")
    data = pd.read_parquet("data/dataset_depth8_Sam_Raph_Sothatsit7.parquet")
    data.pop("game")
    data.pop("rank")
    label = "utility"
    
    print("Splitting data into train and test sets...")
    train_data = data.sample(frac=0.8, random_state=1)
    test_data = data.drop(train_data.index)
    # Convert the data into LightGBM's format
    train_dataset = lgb.Dataset(train_data.drop(
        label, axis=1), label=train_data[label])
    test_dataset = lgb.Dataset(test_data.drop(
        label, axis=1), label=test_data[label])

    # Set the model parameters
    params = {
        "metric": "rmse",
        "objective": "regression",
        'learning_rate': 0.2,
        'num_leaves': 10000,
    }

    def save_often(obj):
        if obj.evaluation_result_list and (obj.iteration + 1) % 100 == 0:
            print("Saving checkpoint to lgb_checkpoint.txt")
            obj.model.save_model('lgb_checkpoint.txt', num_iteration=obj.model.best_iteration)

    print("Training model...")
    # Train the model
    model = lgb.train(params, train_dataset, valid_sets=[test_dataset], num_boost_round=5000, callbacks=[
        save_often,
        lgb.callback.early_stopping(15),
        lgb.callback.log_evaluation()
    ], init_model=lgb.Booster(model_file='lgb_checkpoint.txt'))
    # save model
    model.save_model('lgb_checkpoint.txt', num_iteration=model.best_iteration)

    # Make predictions on the test set
    predictions = model.predict(test_data.drop(label, axis=1))
    print(pd.DataFrame({"pred": predictions, "test": test_data[label]}))


if __name__ == "__main__":
    run()
