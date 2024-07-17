python -m uni2ts.data.builder.simple ETTh1 Datasets/ETTh1.csv --dataset_type wide
python -m uni2ts.data.builder.simple ETTh1 Datasets/ETTh1.csv --date_offset '2017-10-23 23:00:00'
python -m cli.train -cp conf/finetune run_name=example_run model=moirai_1.0_R_small data=etth1 val_data=etth1
python -m cli.eval run_name=example_eval_1 model=moirai_1.0_R_small model.patch_size=32 model.context_length=1000 data=etth1_test
python -m cli.train -cp conf/finetune run_name=swat_run model=moirai_1.0_R_small data=swat val_data=swat

python -m uni2ts.data.builder.simple swat_train Datasets/swat.csv --date_offset '2015-12-27 07:36:00'
python -m cli.train -cp conf/finetune run_name=swat_run model=moirai_1.0_R_small data=swat val_data=swat