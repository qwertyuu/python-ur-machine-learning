import pandas as pd
import pymongo
from src.prep import convert_game_to_cols

def run():
    print("Reading file(s)...")

    df = pd.concat([
        pd.read_csv("data/expectimax_utility_dataset.csv"),
        pd.read_csv("data/expectimax_utility_dataset2.csv"),
        pd.read_csv("data/expectimax_utility_dataset3.csv"),
        pd.read_csv("data/expectimax_utility_dataset4.csv"),
        pd.read_csv("data/expectimax_utility_dataset5.csv"),
    ], ignore_index=True)

    df = convert_game_to_cols(df)

    print(f"Dropping duplicates... Count before:{len(df)}")
    df.drop_duplicates(["game", "x", "y", "roll"], inplace=True)
    print(f"Count after:{len(df)}")
    #df = df.drop(columns=["game"])
    df.to_parquet("data/dataset3.parquet")


if __name__ == "__main__":
    run()