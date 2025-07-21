import os
import pandas as pd

DATA_DIR = 'data'
FOLD = '1'
Y_FILE = 'y_test.parquet'

def main():
    datasets = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    for dataset in datasets:
        y_path = os.path.join(DATA_DIR, dataset, FOLD, Y_FILE)
        if os.path.exists(y_path):
            y = pd.read_parquet(y_path)
            # y could be a DataFrame or Series, handle both
            if hasattr(y, 'dtypes'):
                dtype = y.dtypes
            else:
                dtype = y.dtype
            print(f"{dataset}: {dtype}")
        else:
            print(f"{dataset}: {y_path} not found")

if __name__ == '__main__':
    main()
