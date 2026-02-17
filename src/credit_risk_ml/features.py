TARGET_COL_RAW = "default payment next month"
TARGET_COL = "default"

def preprocess(df):
    df = df.copy()

    # Rename target
    df = df.rename(columns={TARGET_COL_RAW: TARGET_COL})

    # Handle missing values
    df.fillna(0, inplace=True)

    return df