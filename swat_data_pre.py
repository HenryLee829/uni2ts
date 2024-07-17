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
df['Normal/Attack'] = df['Normal/Attack'].replace({"Normal": 0, "Attack": 1, "A ttack": 1})
df[' Timestamp'] = df[' Timestamp'].str.strip()
df[' Timestamp'] = pd.to_datetime(df[' Timestamp'], format='%d/%m/%Y %I:%M:%S %p')
df.set_index(' Timestamp', inplace=True)
pass


def multivar_example_gen_func() -> Generator[dict[str, Any], None, None]:
    yield {
        "target": df.to_numpy().T,  # array of shape (var, time)
        "start": df.index[0],
        "freq": pd.infer_freq(df.index),
        "item_id": "item_0",
    }


features = Features(
    dict(
        target=Sequence(
            Sequence(Value("float32")), length=len(df.columns)
        ),  # multivariate time series are saved as (var, time)
        start=Value("timestamp[s]"),
        freq=Value("string"),
        item_id=Value("string"),
    )
)

hf_dataset = datasets.Dataset.from_generator(
    multivar_example_gen_func, features=features
)
hf_dataset.save_to_disk("swat_dataset_multi")
