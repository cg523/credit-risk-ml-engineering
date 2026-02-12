import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load the credit risk dataset from a CSV file.

    Parameters:
    path (str): The file path to the CSV dataset.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded dataset.
    """

    data = pd.read_csv(path)
    
    return data