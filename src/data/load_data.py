import pandas as pd
import os

def load_data(file_name):
    file_path = os.path.join("data/combined", file_name)
    return pd.read_csv(file_path)