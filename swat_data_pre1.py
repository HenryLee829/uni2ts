from collections.abc import Generator
from pathlib import Path
from typing import Any

import datasets
import pandas as pd
from datasets import Features, Sequence, Value

normal_data_path = 'Datasets/SWaT_A1_A2/SWaT_Dataset_Normal_v0.csv'
attack_data_path = 'Datasets/SWaT_A1_A2/SWaT_Dataset_Attack_v0.csv'
skiprows = 1
steady_row = 21600
df = pd.read_csv(normal_data_path, skiprows=skiprows)[steady_row:]
# df['Normal/Attack'] = df['Normal/Attack'].replace({"Normal": 0, "Attack": 1, "A ttack": 1})
df.drop('Normal/Attack', axis=1, inplace=True)
df[' Timestamp'] = df[' Timestamp'].str.strip()
df[' Timestamp'] = pd.to_datetime(df[' Timestamp'], format='%d/%m/%Y %I:%M:%S %p')
df.rename(columns={' Timestamp': 'date'}, inplace=True)
df.set_index('date', inplace=True)
df.to_csv('Datasets/swat.csv')

# df2 = pd.read_csv('Datasets/swat.csv', skiprows=skiprows)
df1 = pd.read_csv('Datasets/swat.csv')
df3 = pd.read_csv('Datasets/ETTh1.csv')
