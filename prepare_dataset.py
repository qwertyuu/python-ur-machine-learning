import pandas as pd
import glob
import csv
from concurrent.futures import ThreadPoolExecutor


ignore = [3, 7, 11, 15, 16, 18, 19, 20, 22, 23, 27]

col_str_pos_map = {}

col_i = 0
for str_i in range(31):
    if str_i not in ignore:
        col_str_pos_map[col_i] = str_i
        col_i += 1

str_val_map = {
    'D': 1,
    'L': -1,
    '-': 0
}

def split(list_a, chunk_size):
  for i in range(0, len(list_a), chunk_size):
    yield list_a[i:i + chunk_size]


def read(filename):
    with open(filename) as csvfile:
        print(f"Reading {filename}...")
        spamreader = csv.reader(csvfile)
        headers = next(spamreader)
        return {"headers": headers, "data": list(csv.reader(csvfile))}


def work(data):
    final = []
    index = 0
    for row in data:
        for i in range(20):
            row.append(str_val_map[row[0][col_str_pos_map[i]]])
        if index % 10000 == 0:
            print(index/len(data))
        index += 1
        final.append(row)
    print("done")
    return pd.DataFrame(data)

def run():
    with ThreadPoolExecutor(max_workers = 20) as executor:
        results = executor.map(read, glob.glob("data/expectimax_8/*.csv"))


    concat_data = []
    headers = None
    for result in results:
        concat_data.extend(result["data"])
        headers = result["headers"]
    
    with ThreadPoolExecutor(max_workers = 100) as executor:
        results2 = executor.map(work, split(concat_data, 100000))

    df = pd.concat(results2, ignore_index=True)
    df = df.set_axis(headers + ['game' + str(i) for i in range(20)], axis=1, copy=False)
    df = df.astype({
        "roll": int,
        "x": int,
        "y": int,
        "light_turn": bool,
        "light_score": int,
        "dark_score": int,
        "light_left": int,
        "dark_left": int,
        "rank": int,
        "utility": float,
    })
    print(f"Dropping duplicates... Count before:{len(df)}")
    df.drop_duplicates(["game", "x", "y", "roll"], inplace=True)
    print(f"Count after:{len(df)}")
    print(df.head())
    df.to_parquet("data/dataset_depth8_Sam_Raph_Sothatsit6.parquet")


if __name__ == "__main__":
    run()