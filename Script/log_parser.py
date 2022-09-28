'''
python log_parser.py --file [FILE_PATH]
'''
import pandas as pd
import argparse

# argument parsing
parser = argparse.ArgumentParser(description='결과 로그를 레이어번호, 레이어종류, 오답률로 분리한 csv파일을 만듭니다. pandas 설치 필요.')
parser.add_argument('--file', required=True, help='변환할 파일 경로')
args = parser.parse_args()

# open file and parse
file_path = args.file
f = open(file_path, 'r')

rate = []
name = []

for line in f.readlines():
    if line.startswith('Layer #'):
        tokens = line.split()
        rate.append(tokens[-2][:-2])
        name.append(tokens[-1])

rate = rate[1:]
name = name[1:]

# make dataframe and save
data = pd.DataFrame([name, rate]).transpose()
save_dir = file_path.split('.')[0] + '.csv'
data.to_csv(save_dir)
print(f'Result saved at {save_dir}')