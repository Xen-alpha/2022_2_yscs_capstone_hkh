# Vessl.ai Code

## 설명
[vessl.ai](https://vessl.ai) 에서 돌릴 수 있는 코드입니다.   

## 사용법
1. `vessl.ai`에 로그인
2. `Projects`의 `neuron-single-bit-flip` 들어가기
3. `New experiment` 또는 기존의 실험 결과창에 들어가서 `Reproduce` 누르기
4. `Datasets, codes or files`의 `Add Dataset`으로 `/input`에 사용하고픈 데이터셋 넣기
5. `Start command` 및 `Hyperparameters` 적고 `Deploy`

## Start command
```
pip install bitstring && git clone https://github.com/Xen-alpha/2022_2_yscs_capstone_hkh.git && mv 2022_2_yscs_capstone_hkh/Script/vessl_code/* ~ && git clone https://github.com/WaiNaat/pytorchfi.git && python neuron.py
```
만약 자세한 바이너리 로그 파일도 저장하고 싶으시면 `python neuron.py --detailed-log` 사용하시면 됩니다.

## Hyperparameters
| Key | Default value | Description |
|:---:|:-----:|:-----------:|
|model_name|`vgg19_bn`|[chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models)에 있는 모델명 복붙. 5-9번 실험 참고하시면 돌아가는 모델과 아닌 모델 대충 알 수 있습니다|    
|dataset|`cifar10`|`cifar10` 또는 `cifar100`|
|seed|`-1`|default로 냅두면 현재시간 기반 랜덤 시드를 만듭니다. 만들어진 랜덤 시드값도 로그에 저장돼요|
|batch_size|`256`||
|img_size|`32`||
|channels|`3`||
|bit_flip_pos|`-1`|특정 위치에만 bit flip을 일으킬 수 있습니다. 기본값으로 냅두면 무작위에요. 11번 실험 결과 참고|
|layer_type|`all`|띄어쓰기 없는 쉼표 `,` 로 원하는 레이어 종류만 지정이 가능합니다. 3번 실험 참고 |
|layer_num|`all`|띄어쓰기 없는 쉼표 `,` 로 원하는 레이어 번호만 지정이 가능합니다. 10번 실험 참고|

## 다른 모델을 사용하고 싶을 때
[neuron.py#L66](https://github.com/Xen-alpha/2022_2_yscs_capstone_hkh/blob/main/Script/vessl_code/neuron.py#L66)를 원하는 모델로 고치면 됩니다.    
모델을 바꾼 후에는 [neuron.py#L77](https://github.com/Xen-alpha/2022_2_yscs_capstone_hkh/blob/main/Script/vessl_code/neuron.py#L77) 또는 [neuron.py#L87](https://github.com/Xen-alpha/2022_2_yscs_capstone_hkh/blob/main/Script/vessl_code/neuron.py#L87) 값도 맞춰서 바꿔주세요.