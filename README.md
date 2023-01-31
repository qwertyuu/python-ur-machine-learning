# Royal Game of Ur ML Experiments
This project aims to conduct machine learning experiments on the Royal Game of Ur.

## Getting Started
### Prerequisites
Python 3.x
### Setup
Create a virtual environment (optional) using the following command:

`python -m venv venv`

Activate the virtual environment and install the required packages using the command:

`pip install -r requirements.txt`
## Usage

### Data Preparation

The project requires data in the data folder in the form of CSV files.

Adjust the code in prepare_dataset.py to fit your CSV files.

Rename the output dataset file to a suitable name.

Run the following command to prepare the dataset:

`python prepare_dataset.py`

## Training

Edit train.py to use your dataset file.

Run the following command to start the training process:

`python train.py`

The script will overwrite the model file `lgb.pkl` every time it runs, so make sure to backup your model.

## Inference

Edit infer.py to use your trained model file.

Run the following command to start the inference process:

`python infer.py`

The script will spawn a local web server at http://localhost:5000/infer which you can use to make inference requests.

### Inference Request

Make a POST request to the server with the following data structure:

```
[
  {
    "game": "L-- --- --- --- .-. .-. --- ---",
    "roll": 1,
    "x": 0,
    "y": 0,
    "light_turn": true,
    "light_score": 0,
    "dark_score": 0,
    "light_left": 6,
    "dark_left": 7
  },
  {
    "game": "L-- --- --- --- .-. .-. --- ---",
    "roll": 1,
    "x": 0,
    "y": 4,
    "light_turn": true,
    "light_score": 0,
    "dark_score": 0,
    "light_left": 6,
    "dark_left": 7
  }
]
```

Expected response:

```
{
  "utilities": [
    -0.31994735231754157,
    -0.7774157732275543
  ]
}
```