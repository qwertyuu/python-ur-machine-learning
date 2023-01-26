import pandas as pd

def run():
    out2 = pd.read_csv("data/out2.csv")
    out3 = pd.read_csv("data/out3.csv")
    out4 = pd.read_csv("data/out4.csv")
    out5 = pd.read_csv("data/out5.csv")

    final = pd.concat([
        out2,
        out3,
        out4,
        out5
    ], ignore_index=True)

    final.to_parquet("data/final2.parquet")

if __name__ == "__main__":
    run()