import os
import torch
import pandas as pd

os.makedirs(os.path.join("data01"), exist_ok=True)
data_file = os.path.join("data01","house_timy.csv")
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,10600\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file, skipinitialspace=True)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# Fill only numeric columns with their mean
num_cols = inputs.select_dtypes(include='number').columns


# 对分类特征包括 NaN 做 one-hot 编码
inputs = pd.get_dummies(inputs, dummy_na=True, dtype=int)
print(inputs)

