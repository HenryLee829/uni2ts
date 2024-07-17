import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from einops import rearrange
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from uni2ts.distribution import MixtureOutput, StudentTOutput, NormalFixedScaleOutput, NegativeBinomialOutput, \
    LogNormalOutput
from uni2ts.eval_util.plot import plot_single, plot_next_multi
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
PDT = 20  # prediction length: any positive integer
CTX = 200  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32  # batch size: any positive integer
# TEST = 100  # test set length: any positive integer
TEST_RATE = 0.4

normal_data_path = 'Datasets/SWaT_A1_A2/SWaT_Dataset_Normal_v0.csv'
attack_data_path = 'Datasets/SWaT_A1_A2/SWaT_Dataset_Attack_v0.csv'
checkpoint_path = 'outputs/finetune/moirai_1.0_R_small/swat/swat_run/checkpoints/epoch=3-step=400.ckpt'
skiprows = 1
steady_row = 21600
df = pd.read_csv(normal_data_path, skiprows=skiprows)[steady_row:]
# df['Normal/Attack'] = df['Normal/Attack'].replace({"Normal": 0, "Attack": 1, "A ttack": 1})
df.drop('Normal/Attack', axis=1, inplace=True)
df[' Timestamp'] = df[' Timestamp'].str.strip()
df[' Timestamp'] = pd.to_datetime(df[' Timestamp'], format='%d/%m/%Y %I:%M:%S %p')
df.set_index(' Timestamp', inplace=True)

ds = PandasDataset(dict(df))

# grouper = MultivariateGrouper(len(ds))
# multivar_ds = grouper(ds)

TEST = int(len(df) * TEST_RATE)
train, test_template = split(
    ds, offset=-TEST
)  # assign last TEST time steps as test set

# Construct rolling window evaluation
test_data = test_template.generate_instances(
    prediction_length=PDT,  # number of time steps for each prediction
    windows=TEST // PDT,  # number of windows in rolling window evaluation
    distance=PDT,  # number of time steps between each window - distance=PDT for non-overlapping windows
)
# for instance in test_data:
#     pass
# for entry in train:
#     pass

# model = MoiraiForecast(
#     module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{SIZE}"),
#     prediction_length=PDT,
#     context_length=CTX,
#     patch_size=PSZ,
#     num_samples=100,
#     target_dim=len(ds),
#     feat_dynamic_real_dim=ds.num_feat_dynamic_real,
#     past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
# )

# 定义 distr_output
distr_output = MixtureOutput(
    components=[
        StudentTOutput(),
        NormalFixedScaleOutput(),
        NegativeBinomialOutput(),
        LogNormalOutput()
    ]
)

# 指定模型参数
module_kwargs = {
    "distr_output": distr_output,
    "d_model": 384,
    "num_layers": 6,
    "patch_sizes": (8, 16, 32, 64, 128),
    "max_seq_len": 512,
    "attn_dropout_p": 0.0,
    "dropout_p": 0.0,
    "scaling": True
}

model = MoiraiForecast.load_from_checkpoint(
    module_kwargs=module_kwargs,
    prediction_length=PDT,
    context_length=CTX,
    patch_size=PSZ,
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    checkpoint_path=checkpoint_path,
)

predictor = model.create_predictor(batch_size=BSZ)
forecasts = predictor.predict(test_data.input)

input_it = iter(test_data.input)
label_it = iter(test_data.label)
forecast_it = iter(forecasts)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(25, 10))
plot_next_multi(
    axes,
    input_it,
    label_it,
    forecast_it,
    context_length=200,
    intervals=(0.5, 0.9),
    dim=None,
    name="pred",
    show_label=True,
)
plt.show()

# for inp, label, forecast in zip(input_it, label_it, forecast_it):
#     # inp = next(input_it)
#     # label = next(label_it)
#     # forecast = next(forecast_it)
#
#     fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(25, 10))
#     for i, ax in enumerate(axes.flatten()):
#         plot_single(
#             inp,
#             label,
#             forecast,
#             context_length=200,
#             intervals=(0.5, 0.9),
#             dim=i,
#             ax=ax,
#             name="pred",
#             show_label=True,
#         )
#     plt.show()
pass
