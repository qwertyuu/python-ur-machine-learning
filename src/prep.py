import time
import pandas as pd


def replace_chars(x):
    x = x.replace(' ', '').replace('.', '')
    x = x.replace('D', '3').replace('L', '1').replace('-', '2')
    s = list(x)
    return pd.Series({'game' + str(i): s[i] for i in range(20)})


def convert_game_to_cols(df):
    # print("Converting 'game' to a series of [-1, 1] columns...")
    t1 = time.time()
    #j = df['game'].str.replace('\s|\.', '', regex=True).str.translate(table).str.split('', expand=True)
    j = df['game'].apply(replace_chars)
    j = j.astype("int32") - 2
    #print(j)
    k = ['game' + str(i) for i in range(20)]
    df[k] = j
    #print(time.time() - t1)
    return df