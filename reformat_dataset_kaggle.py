import time
import pandas as pd


l = {
    "game0": "A1",
    "game1": "B1",
    "game2": "C1",
    "game3": "A2",
    "game4": "B2",
    "game5": "C2",
    "game6": "A3",
    "game7": "B3",
    "game8": "C3",
    "game9": "A4",
    "game10": "B4",
    "game11": "C4",
    # "game": # "A5",
    "game12": "B5",
    # "game": # "C5",
    # "game": # "A6",
    "game13": "B6",
    # "game": # "C6",
    "game14": "A7",
    "game15": "B7",
    "game16": "C7",
    "game17": "A8",
    "game18": "B8",
    "game19": "C8",
}
str_val_map = {
    1: 'D',
    -1: 'L',
    0: 'E',
}


def replace_chars(x):
    return str_val_map[x]


def convert_game_to_cols(df):
    t1 = time.time()
    for i in range(20):
        df[f'game{i}'] = df[f'game{i}'].apply(replace_chars)
    df["x"].replace([0, 1, 2], ['A', 'B', 'C'], inplace=True)
    df["y"] = df["y"] + 1
    df['light_turn'] = (df['light_turn'] == 'true') * 1
    df = df.rename(columns=l)
    print(time.time() - t1)
    return df


def main():
    dataset = pd.read_parquet("data/dataset_depth8_Sam_Raph_Sothatsit6.parquet")
    new_dataset = convert_game_to_cols(dataset)
    new_dataset.to_csv("data/dataset_depth8_Sam_Raph_Sothatsit6_Kaggle.csv", index=False)
    new_dataset.to_parquet("data/dataset_depth8_Sam_Raph_Sothatsit6_Kaggle.parquet")


if __name__ == '__main__':
    main()
