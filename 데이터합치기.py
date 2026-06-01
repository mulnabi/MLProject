import pandas as pd
import glob

name=input('합칠 파일 폴더명:')

path = name+'/*.csv'
file_list = glob.glob(path)
data = pd.concat([pd.read_csv(file) for file in file_list], axis=0)
data.to_csv(f'#{name}.csv', index=False, encoding='utf-8-sig')