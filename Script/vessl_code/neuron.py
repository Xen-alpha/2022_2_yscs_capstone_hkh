import torch
import torchvision
import random
import copy
import numpy as np
import pandas as pd
import os
import argparse
import datetime
import vessl

from torchvision import transforms

from base_fault_injection import add_input_layer, neuron_single_bit_flip

import pytorchfi # git clone https://github.com/WaiNaat/pytorchfi.git
from pytorchfi.core import FaultInjection
from pytorchfi.neuron_error_models import random_neuron_location

vessl.init()

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorchFI single bit flip example code for vessl.ai')
parser.add_argument('--input-path', type=str, default='/input', help='input files path')
parser.add_argument('--output-path', type=str, default='/output', help='output files path')
parser.add_argument('--detailed-log', action='store_true', default=False, help='Save detailed single bit flip log')
args = parser.parse_args()

# vessl.ai hyperparameters
model_name =  str(os.environ.get('model_name', 'vgg19_bn')) # model names at https://github.com/chenyaofo/pytorch-cifar-models
dataset = str(os.environ.get('dataset', 'cifar10'))
seed = int(os.environ.get('seed', -1))
batch_size = int(os.environ.get('batch_size', 256))
img_size = int(os.environ.get('img_size', 32))
channels = int(os.environ.get('channels', 3))
bit_flip_pos = int(os.environ.get('bit_flip_pos', -1))
layer_type = str(os.environ.get('layer_type', 'all'))
layer_nums = str(os.environ.get('layer_nums', 'all'))

if seed < 0:
    seed = int(datetime.datetime.now().timestamp())

if bit_flip_pos < 0:
    bit_flip_pos = None

if layer_type != 'all':
    layer_type = list(map(lambda x: getattr(torch.nn, x), layer_type.split(',')))
else:
    layer_type = ['all']

# gpu setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')
print(f'Device count: {torch.cuda.device_count()}')

# seed setting (https://hoya012.github.io/blog/reproducible_pytorch/)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

# load model
model = torch.hub.load("chenyaofo/pytorch-cifar-models", dataset + '_' + model_name, pretrained=True)
model = add_input_layer(model)
model.to(device)

# preprocess data
# Transform statics from https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/vgg11_bn/default.log
dataloader = None
if dataset == 'cifar10':
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
        ]
    )
    data = torchvision.datasets.CIFAR10(root=args.input_path, train=False, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

elif dataset == 'cifar100':
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
        ]
    )
    data = torchvision.datasets.CIFAR100(root=args.input_path, train=False, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

else:
    raise AssertionError(f'Invalid dataset name {dataset}')

# make fault injection base model
base_fi_model = neuron_single_bit_flip(
    model = copy.deepcopy(model),
    batch_size = batch_size, 
    input_shape = [channels, img_size, img_size],
    layer_types = layer_type,
    flip_bit_pos = bit_flip_pos,
    save_log_list = args.detailed_log
)

print(base_fi_model.print_pytorchfi_layer_summary())

# fault injection layer range setting
if layer_nums != 'all':
    layer_nums = list(map(int, layer_nums.split(',')))
    layer_nums.sort()
    while layer_nums and layer_nums[-1] >= base_fi_model.get_total_layers():
        layer_nums.pop()
else:
    layer_nums = range(base_fi_model.get_total_layers())

# experiment
print(f'Seed: {seed}')
results = []
misclassification_rate = []
layer_name = []
error_logs = []

for layer_num in layer_nums:
    
    orig_correct_cnt = 0
    orig_corrupt_diff_cnt = 0
    batch_idx = -1
    
    for images, labels in dataloader:

        batch_idx += 1

        images = images.to(device)

        # original model inference
        model.eval()
        with torch.no_grad():
            orig_output = model(images)

        # determine single bit flip position
        layer_num_list = []
        dim1 = []
        dim2 = []
        dim3 = []

        for _ in range(batch_size):
            layer, C, H, W = random_neuron_location(base_fi_model, layer=layer_num)

            layer_num_list.append(layer)
            dim1.append(C)
            dim2.append(H)
            dim3.append(W)

        # make corrupted model
        base_fi_model.reset_log()
        corrupted_model = base_fi_model.declare_neuron_fault_injection(
            batch = [i for i in range(batch_size)],
            layer_num = layer_num_list,
            dim1 = dim1,
            dim2 = dim2,
            dim3 = dim3,
            function = base_fi_model.single_bit_flip_function
        )

        # corrupted model inference
        corrupted_model.eval()
        with torch.no_grad():
            corrupted_output = corrupted_model(images)

        # get label
        original_output = torch.argmax(orig_output, dim=1).cpu().numpy()
        corrupted_output = torch.argmax(corrupted_output, dim=1).cpu().numpy()
        labels = labels.numpy()

        # calc result
        for i in range(batch_size):

            if labels[i] == original_output[i]:
                orig_correct_cnt += 1

                if original_output[i] != corrupted_output[i]:
                    orig_corrupt_diff_cnt += 1

                    if args.detailed_log:
                        log = [
                            f'Layer: {layer_num}',
                            f'Batch: {batch_idx}',
                            f'Position: ({i}, {dim1[i]}, {dim2[i]}, {dim3[i]})',
                            f'Original value:  {base_fi_model.log_original_value[i]}',
                            f'Original binary: {base_fi_model.log_original_value_bin[i]}',
                            f'Flip bit: {base_fi_model.log_bit_pos[i]}',
                            f'Error value:     {base_fi_model.log_error_value[i]}',
                            f'Error binary:    {base_fi_model.log_error_value_bin[i]}',
                            f'Label:        {labels[i]}',
                            f'Model output: {corrupted_output[i]}',
                            '\n'
                        ]

                        error_logs.append('\n'.join(log))

    # save result
    rate = orig_corrupt_diff_cnt / orig_correct_cnt * 100
    result = f'Layer #{layer_num}: {orig_corrupt_diff_cnt} / {orig_correct_cnt} = {rate:.4f}%, ' + str(base_fi_model.layers_type[layer_num]).split(".")[-1].split("'")[0]
    print(result)
    results.append(result)
    misclassification_rate.append(rate)
    layer_name.append(str(base_fi_model.layers_type[layer_num]).split(".")[-1].split("'")[0])
    vessl.log(step=layer_num, payload={'Misclassification_rate': rate})

# save log file
# save overall log
save_path = os.path.join(args.output_path, '_'.join([model_name, dataset, str(seed)]))
vessl.log({'seed': seed})

f = open(save_path + '.txt', 'w')

f.write(base_fi_model.print_pytorchfi_layer_summary())
f.write(f'\n\n===== Result =====\nSeed: {seed}\nSpecific bit flip position: {bit_flip_pos}\n')
for result in results:
    f.write(result + '\n')

f.close()

# save detailed log
if args.detailed_log:
    f = open(save_path + '_detailed.txt', 'w')

    for error_log in error_logs:
        f.write(error_log + '\n')

    f.close()

# save misclassification rate
data = pd.DataFrame([layer_name, misclassification_rate]).transpose()
data.to_csv(save_path + '.csv')