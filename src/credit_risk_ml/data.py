import pandas as pd

TARGET_COL_RAW = "default payment next month"
TARGET_COL = "default"

def load_data(path: str) -> pd.DataFrame:
    """
    Load the credit risk dataset from a CSV file.

    Parameters:
    path (str): The file path to the CSV dataset.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded dataset.
    """

    df = pd.read_csv(path, header=1)
    df = df.rename(columns={TARGET_COL_RAW: TARGET_COL})


    return df