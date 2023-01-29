#import lightgbm as lgb
import pandas as pd
import optuna
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from sklearn.model_selection import RepeatedKFold

data = pd.read_parquet("data/dataset.parquet")
train_data = data.sample(frac=0.8, random_state=1)
test_data = data.drop(train_data.index)
# Convert the data into LightGBM's format


def objective(trial):
    # Set the model parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 1.0),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        "early_stopping_round": 10,
        #'num_iterations': trial.suggest_int('num_iterations', 50, 1000),
    }
    train_dataset = lgb.Dataset(train_data.drop('utility', axis=1), label=train_data['utility'])
    test_dataset = lgb.Dataset(test_data.drop('utility', axis=1), label=test_data['utility'])

    # Train and evaluate the model
    model = lgb.train(params, train_dataset, valid_sets=[test_dataset], num_boost_round=100, verbose_eval=50)
    predictions = model.predict(test_data.drop('utility', axis=1))
    mae = mean_absolute_error(test_data['utility'], predictions)

    return mae

study = optuna.create_study(direction='minimize')
try:
    study.optimize(objective, n_trials=50, n_jobs=2)
finally:
    print(study.best_params)
    print(study.best_value)