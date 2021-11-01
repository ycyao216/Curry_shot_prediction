import pandas as pd
import numpy as np

# Initalize variables
data_path = 'shot_logs.csv'
player = 'stephen curry'
scaling_features = ['SHOT_DIST']
features_path = 'features.csv'
labels_path = 'labels.csv'

def process_data(data_path, player, scaling_features, feature_path, labels_path):

    # Load data and select player
    df = pd.read_csv(data_path)
    df = df[df['player_name'] == player]

    # Select features and scale
    for col in scaling_features:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    # Export features and labels to csv
    df[scaling_features].to_csv(feature_path, index=False)
    df['SHOT_RESULT'].replace({'missed': 0, 'made': 1}).to_csv(labels_path, index=False)

if __name__ == '__main__':
    process_data(data_path, player, scaling_features, features_path, labels_path)