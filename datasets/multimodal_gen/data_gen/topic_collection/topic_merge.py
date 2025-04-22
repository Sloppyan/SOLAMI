import os
import pandas as pd


data_path = "SOLAMI/datasets/multimodal_gen/data/topics/merged_topics.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(df.head())
    data_items = df.to_dict(orient='records')
    print(len(data_items))
    
pass