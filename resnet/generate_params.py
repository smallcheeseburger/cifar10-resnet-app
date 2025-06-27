import json
import os

params_list = [
    {'lr': 0.001, 'batch_size': 128},
    {'lr': 0.001, 'batch_size': 256},
    {'lr': 0.0005, 'batch_size': 128},
    {'lr': 0.0005, 'batch_size': 256},
]


os.makedirs('params', exist_ok=True)
# Generate parameters
for i, params in enumerate(params_list):
    with open(f'params/params_{i}.json', 'w') as f:
        json.dump(params, f, indent=4)

print("Successfully generate parameters")