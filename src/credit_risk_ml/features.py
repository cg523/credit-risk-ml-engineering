def preprocess(df):
    df = df.copy()
    df.fillna(0, inplace=True)
    return df