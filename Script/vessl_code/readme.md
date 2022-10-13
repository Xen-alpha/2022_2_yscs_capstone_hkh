# Vessl.ai Code

## 설명
[vessl.ai](https://vessl.ai) 에서 돌릴 수 있는 코드입니다.   

## 사용법
### 기본 실험 돌리기
1. `vessl.ai`에 로그인
2. `Projects`에서 아무데나 들어가기 또는 `New project` 누른 후 이름만 정하고 `Create`
3. `New experiment` 누르고 Docker image를 `Python 3.7 (All Packages, CUDA 11.2)`로 변경
4. `Add Dataset`으로 `/input`에 사용하고픈 데이터셋 넣기
5. `Start command` 및 `Hyperparameters` 적고 `Deploy`
6. `Logs`와 `Plots`에서 실시간으로 결과 확인 가능. 최종 결과 파일은 실험 종료 후 `Files`에 저장됨

### Sweep으로 seed만 바꿔서 여러번 자동으로 돌리기
1. 프로젝트의 `Sweeps` &rightarrow; `New sweep` 누르기
2. Type: `아무거나`, Goal: `대충 높은숫자`, Metric: `Misclassification_rate`
3. Max experiment count: `연속으로 돌릴 횟수`, Parallel experiment count: `1`, Max failed experiment count: `연속으로 돌릴 횟수`
4. Algorithm name: `random`
5. `Add Parameter` 누르고 Name: `seed`, Type: `int`, Range type: `Search space`, Value `1` 부터 `2147483647`까지
6. Runtime은 기존 실험 베껴오거나 위의 `기본 실험 돌리기`와 같은 방법으로 설정하되 hyperparameters에 `seed`는 입력하지 않음
7. `Deploy`후 놀다오기

## Start command

### neuron
```
pip install bitstring && git clone https://github.com/Xen-alpha/2022_2_yscs_capstone_hkh.git && mv 2022_2_yscs_capstone_hkh/Script/vessl_code/* ~ && git clone https://github.com/WaiNaat/pytorchfi.git && python neuron.py
```
### weight
```
pip install bitstring && git clone https://github.com/Xen-alpha/2022_2_yscs_capstone_hkh.git && mv 2022_2_yscs_capstone_hkh/Script/vessl_code/* ~ && git clone https://github.com/WaiNaat/pytorchfi.git && python weight.py
```
만약 자세한 바이너리 로그 파일도 저장하고 싶으시면 `python neuron.py --detailed-log` 사용하시면 됩니다.      
나는 정말 모르겠다 &leftarrow; 아무 실험 들어간뒤 오른쪽 위 `Reproduce` 눌러서 똑같이 베끼고 hyperparameter들 바꿔주시면 됩니다.

## Hyperparameters
대표적인 예시는 `neuron-single-bit-flip` 프로젝트에 있습니다.    

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

weight fault injection은 26번과 27번 실험 참고