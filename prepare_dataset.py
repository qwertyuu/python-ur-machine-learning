import pandas as pd
from src.prep import convert_game_to_cols

def run():
    print("Reading file(s)...")

    df = pd.concat([
        pd.read_csv("data/expectimax_8/expectimax8_utility_dataset_sam.csv"),
        pd.read_csv("data/expectimax_8/expectimax8_utility_dataset_sam1.csv"),
        pd.read_csv("data/expectimax_8/expectimax8_utility_dataset_sam2.csv"),
        pd.read_csv("data/expectimax_8/expectimax8_utility_dataset_sam3.csv"),
        pd.read_csv("data/expectimax_8/expectimax8_utility_dataset_sam4.csv"),
        pd.read_csv("data/expectimax_8/expectimax8_utility_dataset.csv"),
        pd.read_csv("data/expectimax_8/expectimax8_utility_dataset2.csv"),
        pd.read_csv("data/expectimax_8/expectimax8_utility_dataset3.csv"),
        pd.read_csv("data/expectimax_8/expectimax8_utility_dataset4.csv"),
        pd.read_csv("data/expectimax_8/part1_expectimax8_utility_dataset.csv"),
        pd.read_csv("data/expectimax_8/part2_expectimax8_utility_dataset.csv"),
    ], ignore_index=True).query('rank == rank').astype({
        "roll": int,
        "x": int,
        "y": int,
        "light_score": int,
        "dark_score": int,
        "light_left": int,
        "dark_left": int,
        "rank": int,
    })

    print("Converting 'game' to a series of [-1, 1] columns...")
    df = convert_game_to_cols(df)

    print(f"Dropping duplicates... Count before:{len(df)}")
    df.drop_duplicates(["game", "x", "y", "roll"], inplace=True)
    print(f"Count after:{len(df)}")
    df.to_parquet("data/dataset_depth8_Sam_Raph_Sothatsit3.parquet")


if __name__ == "__main__":
    run()