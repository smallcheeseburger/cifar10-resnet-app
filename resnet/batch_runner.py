import os

import os

params_dir = 'params'

all_files = os.listdir(params_dir)

for params_file in all_files:
    print(f"Running {params_file} ...")
    os.system(f'python resnet/train.py --mode experiment --params {params_file}')