import pandas as pd


def run():
    print("Loading data...")
    data = pd.read_parquet("data/dataset_depth8_Sam_Raph_Sothatsit7_Kaggle.parquet")
    data["x"] = data["x"].replace({"A": 0, "B": 1, "C": 2})
    print("Converting 'columns' to a series of [-1, 0, 1] columns...")
    for col in ["A", "B", "C"]:
        for row in range(1, 9):
            print(f"Converting {col}{row}")
            if f"{col}{row}" not in data.columns:
                continue
            data[f"{col}{row}"] = data[f"{col}{row}"].replace({"L": -1, "E": 0, "D": 1})
    
    # write to parquet
    data.to_parquet("data/dataset_depth8_Sam_Raph_Sothatsit7.parquet")
    


if __name__ == "__main__":
    run()
