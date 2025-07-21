import os
import pandas as pd

def load_tabular_data(input_path):
    """
    Loads a tabular data file from input_path and returns a tuple: (dataframe, dataframe_name).
    Supported formats: .csv, .parquet, .xlsx
    """
    ext = os.path.splitext(input_path)[-1].lower()
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    if ext == '.csv':
        df = pd.read_csv(input_path)
    elif ext == '.parquet':
        df = pd.read_parquet(input_path)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(input_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return df, base_name
