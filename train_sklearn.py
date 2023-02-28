import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def run():
    # Load the data from CSV file
    data = pd.read_parquet('data/dataset_depth8_Sam_Raph_Sothatsit.parquet')
    
    train_data = data.sample(frac=0.8, random_state=1)
    test_data = data.drop(train_data.index)

    # Convert categorical variables into numerical using LabelEncoder
    le = LabelEncoder()
    #train_data['game'] = le.fit_transform(train_data['game'])
    train_data['light_turn'] = le.fit_transform(train_data['light_turn'])

    # Split the data into input (X) and output (y) variables
    X = train_data.drop('utility', axis=1)
    X = X.drop('game', axis=1)
    X = X.drop('rank', axis=1)
    y = train_data['utility']

    # Scale the input data using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Define the MLPRegressor model with appropriate hyperparameters
    model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000)

    # Fit the model on the data
    model.fit(X, y)

    joblib.dump(model, "mlpregressor.pkl")

    # Predict the output for a new input data point
    test_data.pop("game")
    test_data.pop("rank")
    test_utility = test_data.pop("utility")
    new_data = test_data
    #new_data['game'] = le.transform(new_data['game'])
    new_data['light_turn'] = le.transform(new_data['light_turn'])
    new_data = scaler.transform(new_data)
    prediction = model.predict(new_data)

    print(prediction, test_utility)

if __name__ == "__main__":
    run()
