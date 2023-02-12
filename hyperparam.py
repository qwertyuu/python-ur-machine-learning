#import lightgbm as lgb
import pandas as pd
import optuna
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

data = pd.read_parquet("data/dataset_depth8_Sam_Raph_Sothatsit5.parquet")
data.pop("game")
train_data = data.sample(frac=0.8, random_state=1)
test_data = data.drop(train_data.index)


def objective(trial):
    # Set the model parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 100, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
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
    model = lgb.train(params, train_dataset, valid_sets=[test_dataset], num_boost_round=100)
    predictions = model.predict(test_data.drop('utility', axis=1))
    return mean_squared_error(test_data['utility'], predictions)

storage_name = "sqlite:///studies.db"
study = optuna.create_study(study_name="depth_8_7", storage=storage_name, direction='minimize')
try:
    study.optimize(objective, n_trials=1000, n_jobs=2)
finally:
    print(study.best_params)
    print(study.best_value)