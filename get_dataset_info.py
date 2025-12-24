import pandas as pd
import os

file_path = r'c:\DetectionGuard\data\cicids2017_cleaned.csv'
output_file = r'c:\DetectionGuard\dataset_stats.txt'

if os.path.exists(file_path):
    df_head = pd.read_csv(file_path, nrows=5)
    columns = df_head.columns.tolist()
    num_features = len(columns) - 1
    
    with open(file_path, 'rb') as f:
        num_lines = sum(1 for _ in f)
    total_samples = num_lines - 1
    
    df_target = pd.read_csv(file_path, usecols=['Attack Type'])
    class_dist = df_target['Attack Type'].value_counts()
    
    with open(output_file, 'w') as f:
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Number of Features: {num_features}\n")
        f.write("Class Distribution:\n")
        f.write(class_dist.to_string())
        f.write("\n")
    print("Stats saved to dataset_stats.txt")
else:
    print("Dataset file not found.")
