'''
프로토타입

robust_model := single bit flip의 영향을 덜 받도록 고친 모델
망가진 값 := single bit flip으로 인해 망가진 값

생각
1. 망가진 값 자체를 건드릴 경우
1.1. 현재 single_bit_flip_model은 자체적으로 망가진 값과 그 위치를 저장함
    (프로그램 실행 시 --detailed-log를 썼다는 가정 하에)
1.2. 따라서 정상 모델을 돌릴 때 해당 위치에 망가진 값을 삽입하는 방식으로 같은 결과를 얻을 수 있음
1.3. corrupted_model 제작 -> 인퍼런스 -> 망가진 값 수정 -> robust_model에 수정한 값 삽입해서 인퍼런스

2. 특정 레이어 자체를 바꿔버릴 경우
2.1. 만약 weight가 없는 모듈들을 아예 다른 모듈로 교체하는 방식으로 error correction을 할 경우
    https://stackoverflow.com/questions/58297197/how-to-change-activation-layer-in-pytorch-pretrained-module
    이거 참고해서 robust_model 만듦
2.2. 하지만 망가진 값을 가져오는건 안되고 위치만 가져와야함
2.3. single_bit_flip_model에서 deque으로 bit flip pos 지정할수있게 수정하기
        flip_bit_pos=None이고 flip_bit_pos_deque!=None이면 하나씩 빼와서 flip_pos 대체하는 방법?

3. 특정 레이어 아웃풋을 건드릴 경우
(순서주의)
3.1. corrupted_model이랑 똑같은 모델을 만들어서 robust_model이라 이름붙임
3.2. robust_model에 forward hook 걸어서 특정 레이어의 아웃풋을 건드리도록 함

이 코드에선 2번 방법으로
vgg19_bn의 relu를 relu6로 변환
'''

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
from collections import deque

from base_fault_injection import add_input_layer, single_bit_flip_model

import pytorchfi # git clone https://github.com/WaiNaat/pytorchfi.git
from pytorchfi.core import FaultInjection
from pytorchfi.neuron_error_models import random_neuron_location

vessl.init()

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorchFI single bit flip example code for vessl.ai')
parser.add_argument('--input-path', type=str, default='/input', help='input files path')
parser.add_argument('--output-path', type=str, default='/output', help='output files path')
parser.add_argument('--detailed-log', action='store_true', default=True, help='Save detailed single bit flip log')
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

print(model, end='\n\n')

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
base_fi_model = single_bit_flip_model(
    model = copy.deepcopy(model),
    batch_size = batch_size, 
    input_shape = [channels, img_size, img_size],
    layer_types = layer_type,
    flip_bit_pos = bit_flip_pos,
    save_log_list = args.detailed_log
)

print(base_fi_model.print_pytorchfi_layer_summary(), end='\n\n')

# make robust model
# change relu to relu6
def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_layers(module, old, new)
            
        if isinstance(module, old):
            setattr(model, n, new(inplace=module.inplace)) # inplace때문에 relu대상만 가능?

robust_model = copy.deepcopy(model)
replace_layers(robust_model, torch.nn.ReLU, torch.nn.ReLU6)

# make fault injection base for robust model
base_fi_robust_model = single_bit_flip_model(
    model = copy.deepcopy(robust_model),
    batch_size = batch_size, 
    input_shape = [channels, img_size, img_size],
    layer_types = layer_type,
    flip_bit_pos = bit_flip_pos,
    save_log_list = args.detailed_log
)

print(base_fi_robust_model.print_pytorchfi_layer_summary(), end='\n\n')

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
    corrupt_correct_cnt = 0
    robust_correct_cnt = 0
    orig_corrupt_diff_cnt = 0
    orig_robust_diff_cnt = 0
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
            function = base_fi_model.neuron_single_bit_flip_function
        )

        # corrupted model inference
        corrupted_model.eval()
        with torch.no_grad():
            corrupted_output = corrupted_model(images)

        # make robust model
        base_fi_robust_model.reset_log()
        base_fi_robust_model.flip_bit_pos_deque = deque(base_fi_model.log_bit_pos)
        robust_model = base_fi_robust_model.declare_neuron_fault_injection(
            batch = [i for i in range(batch_size)],
            layer_num = layer_num_list,
            dim1 = dim1,
            dim2 = dim2,
            dim3 = dim3,
            function = base_fi_robust_model.neuron_single_bit_flip_function
        )

        # robust model inference
        robust_model.eval()
        with torch.no_grad():
            robust_output = robust_model(images)

        # get label
        original_output = torch.argmax(orig_output, dim=1).cpu().numpy()
        corrupted_output = torch.argmax(corrupted_output, dim=1).cpu().numpy()
        robust_output = torch.argmax(robust_output, dim=1).cpu().numpy()
        labels = labels.numpy()

        # calc result
        for i in range(batch_size):

            if original_output[i] == corrupted_output[i]:
                corrupt_correct_cnt += 1

            if original_output[i] == robust_output[i]:
                robust_correct_cnt += 1

            if labels[i] == original_output[i]:
                orig_correct_cnt += 1

                if original_output[i] != corrupted_output[i]:
                    orig_corrupt_diff_cnt += 1

                if original_output[i] != robust_output[i]:
                    orig_robust_diff_cnt += 1

                if args.detailed_log and robust_output[i] != corrupted_output[i]:
                        log = [
                            f'Layer: {layer_num}',
                            f'Batch: {batch_idx}',
                            f'Position: ({i}, {dim1[i]}, {dim2[i]}, {dim3[i]})',
                            f'Original value:  {base_fi_model.log_original_value[i]}',
                            f'Original binary: {base_fi_model.log_original_value_bin[i]}',
                            f'Flip bit: {base_fi_model.log_bit_pos[i]}',
                            f'Error value:     {base_fi_model.log_error_value[i]}',
                            f'Error binary:    {base_fi_model.log_error_value_bin[i]}',
                            f'Original value(robust model):  {base_fi_robust_model.log_original_value[i]}',
                            f'Original binary(robust model): {base_fi_robust_model.log_original_value_bin[i]}',
                            f'Error value(robust model):     {base_fi_robust_model.log_error_value[i]}',
                            f'Error binary(robust model):    {base_fi_robust_model.log_error_value_bin[i]}',
                            f'Label:        {labels[i]}',
                            f'Corrupted model output: {corrupted_output[i]}',
                            f'Robust model output: {robust_output[i]}'
                            '\n'
                        ]

                        error_logs.append('\n'.join(log))

    # save result
    total_imgs = (batch_idx + 1) * batch_size
    acc_orig = orig_correct_cnt / total_imgs * 100
    acc_corrupt = corrupt_correct_cnt / total_imgs * 100
    acc_robust = robust_correct_cnt / total_imgs * 100
    rate = orig_corrupt_diff_cnt / orig_correct_cnt * 100
    rate2 = orig_robust_diff_cnt / orig_correct_cnt * 100

    result = f'Layer #{layer_num}:'
    result += f'\n    {orig_corrupt_diff_cnt} / {orig_correct_cnt} = {rate:.4f}%, ' + str(base_fi_model.layers_type[layer_num]).split(".")[-1].split("'")[0]
    result += f'\n    {orig_robust_diff_cnt} / {orig_correct_cnt} = {rate2:.4f}%, ' + str(base_fi_robust_model.layers_type[layer_num]).split(".")[-1].split("'")[0]
    result += f'\n    Accuracy: Original {acc_orig:.2f}%, Corrupt {acc_corrupt:.2f}%, Robust {acc_robust:.2f}'
    print(result)

    results.append(result)
    misclassification_rate.append(rate)
    layer_name.append(str(base_fi_model.layers_type[layer_num]).split(".")[-1].split("'")[0])
    vessl.log(step=layer_num, payload={'Misclassification_rate_corrupted_model': rate})
    vessl.log(step=layer_num, payload={'Misclassification_rate_robust_model': rate2})
    vessl.log(step=layer_num, payload={'test': [rate, rate2]})

# save log file
# save overall log
save_path = os.path.join(args.output_path, '_'.join(['neuron', model_name, dataset, str(seed)]))
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
data = pd.DataFrame({'name': layer_name, f'seed_{seed}': misclassification_rate})
data.to_csv(save_path + '.csv')

# save avg value
for name in set(layer_name):
    avg_val = data[data['name'] == name][f'seed_{seed}'].mean()
    vessl.log({name: avg_val})