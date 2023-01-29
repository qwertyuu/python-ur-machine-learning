
def convert_game_to_cols(df):
    print("Converting 'game' to a series of [-1, 1] columns...")
    j = df['game'].str.replace('\s|\.', '', regex=True).str.replace('D', '3').str.replace('L', '1').str.replace('-', '2').str.split('', expand=True)
    j = j.drop([0, 21], axis=1).astype("int32") - 2
    k = ['game' + str(i) for i in range(20)]
    df[k] = j
    return df