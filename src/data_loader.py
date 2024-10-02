import pandas as pd

# Function to load train and test datasets
def load_data(train_path, test_path):

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    columns_to_drop = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
    train.drop(columns=columns_to_drop, axis=1, inplace=True)
    test.drop(columns=columns_to_drop, axis=1, inplace=True)
    
    return train, test
