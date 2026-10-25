import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    data_path = 'tourism_project/data/tourism.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Dataset not found at {data_path}')

    df = pd.read_csv(data_path)

    # Drop completely unnecessary or unique ID columns
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])

    # Handle minimal preprocessing or cleaning: drop rows where target 'ProdTaken' is missing
    if 'ProdTaken' in df.columns:
        df = df.dropna(subset=['ProdTaken'])

    # Split data into Train and Test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['ProdTaken'] if 'ProdTaken' in df.columns else None)

    # Save local train/test splits as CSVs
    train_df.to_csv('Xtrain.csv', index=False) # Simplified names as expected by the workflow artifact step
    test_df.to_csv('Xtest.csv', index=False)

    # Also save with standard expected split names mapping features and target for the training step
    target_col = 'ProdTaken'
    if target_col in df.columns:
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        X_train.to_csv('Xtrain.csv', index=False)
        X_test.to_csv('Xtest.csv', index=False)
        y_train.to_csv('ytrain.csv', index=False)
        y_test.to_csv('ytest.csv', index=False)

    print('--- Data Preparation Completed ---')
    print(f'Train shape: {train_df.shape}, Test shape: {test_df.shape}')

if __name__ == '__main__':
    prepare_data()
