import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import joblib
import math

def run():
    data = pd.read_parquet("data/dataset.parquet")
    train_data = data.sample(frac=0.8, random_state=1)
    test_data = data.drop(train_data.index)
    # Convert the data into LightGBM's format
    train_dataset = lgb.Dataset(train_data.drop('utility', axis=1), label=train_data['utility'])
    test_dataset = lgb.Dataset(test_data.drop('utility', axis=1), label=test_data['utility'])

    # Set the model parameters
    params = {"num_iterations": 5000, "learning_rate": 0.4, "metric": "rmse", "objective": "regression"}

    # Train the model
    model = lgb.train(params, train_dataset, valid_sets=[test_dataset])
    # save model
    joblib.dump(model, 'lgb.pkl')

    # Make predictions on the test set
    predictions = model.predict(test_data.drop('utility', axis=1))
    print(test_data.drop('utility', axis=1))
    print(predictions)
    # Measure the performance of the predictions
    mse = mean_squared_error(test_data['utility'], predictions)

    print(f'rmse: {math.sqrt(mse)}')

if __name__ == "__main__":
    run()