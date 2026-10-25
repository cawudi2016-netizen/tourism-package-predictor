import os
import pandas as pd

def register_data():
    data_path = 'tourism_project/data/tourism.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Dataset not found at {data_path}')

    df = pd.read_csv(data_path)
    print('--- Dataset Registered Successfully ---')
    print(f'Dataset Shape: {df.shape}')
    print('\nColumns and Missing Values:')
    print(df.isnull().sum())

if __name__ == '__main__':
    register_data()
