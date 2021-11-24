import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Initalize variables
data_path = 'shot_logs.csv'
player = 'stephen curry'
scaling_features = ['SHOT_DIST', 'SHOT_NUMBER', 'SHOT_CLOCK', 'TOUCH_TIME']
one_hot_features = []
train_path = 'train4.csv'
valid_path = 'valid4.csv'
test_path = 'test4.csv'

def process_data(data_path, player, scaling_features, one_hot_features, train_path, valid_path, test_path):

    # Load data and select player
    df = pd.read_csv(data_path)
    df = df[df['player_name'] == player]
    
    df = df.fillna(df.mean())

    # Select features and scale
    for col in scaling_features:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    one_hot_features2 = []
    for col in one_hot_features:
        y = pd.get_dummies(df[col])
        df = pd.concat([df, y], axis=1)
        one_hot_features2 += list(y.columns)

    # Encode labels to Missed = 0 and Made = 1
    df['SHOT_RESULT'] = df['SHOT_RESULT'].replace({'missed': 0, 'made': 1})

    # Split to train, validation, and test data with a 80/10/10 split
    train, test = train_test_split(df, test_size=0.1, random_state=42)
    train, valid = train_test_split(train, test_size=1/9, random_state=42)

    # Select columns for output
    cols = scaling_features
    cols += one_hot_features2
    cols.append('SHOT_RESULT')

    # Export to csv
    train[cols].to_csv(train_path, index=False)
    valid[cols].to_csv(valid_path, index=False)
    test[cols].to_csv(test_path, index=False)

if __name__ == '__main__':
    process_data(data_path, player, scaling_features, one_hot_features, train_path, valid_path, test_path)