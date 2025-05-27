# python切片

![image-20250517211953251](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20250517211953251.png)

# tensor.size()、shape()、numel()

### tensor.size()查看张量的维度信息（shape()与他相同）

```python
import torch
a = torch.tensor([[ 0.0349,  0.0670, -0.0612, 0.0280, -0.0222,  0.0422],
         [-1.6719,  0.1242, -0.6488, 0.3313, -1.3965, -0.0682],
         [-1.3419,  0.4485, -0.6589, 0.1420, -0.3260, -0.4795],
         [-0.0658, -0.1490, -0.1684, 0.7188,  0.3129, -0.1116],
         [-0.2098, -0.2980,  0.1126, 0.9666, -0.0178,  0.1222],
         [ 0.1179, -0.4622, -0.2112, 1.1151,  0.1846,  0.4283]])
# 查看tensor的维度信息：torch.Size([6, 6])
print(a.size())
```

### numel()是“number of element”的简写。numel()可以直接返回int类型的元素个数

```python
import torch
 
a = torch.tensor([[ 0.0349,  0.0670, -0.0612, 0.0280, -0.0222,  0.0422],
         [-1.6719,  0.1242, -0.6488, 0.3313, -1.3965, -0.0682],
         [-1.3419,  0.4485, -0.6589, 0.1420, -0.3260, -0.4795],
         [-0.0658, -0.1490, -0.1684, 0.7188,  0.3129, -0.1116],
         [-0.2098, -0.2980,  0.1126, 0.9666, -0.0178,  0.1222],
         [ 0.1179, -0.4622, -0.2112, 1.1151,  0.1846,  0.4283]])
b = a.numel()
print(type(b)) # int
print(b) # 36
```

# 对矩阵进行解析

![image-20250517220036842](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20250517220036842.png)

对于上图的解释：其结构可以理解为包含**两个维度为3 * 4的二维矩阵。 **

**可视化提示：绘制单个3*4的矩阵，然不过他们两个堆叠起来** 

# torch.exp()、torch.pow()

# torch.arange(12, dtype=torch.dloat32).reshape((3,4))

# pandas

```python
import os

os.makedirs(os.path.join("code","study"),exit_ok=True)

os.makedirs(os.path.join("code", "study01", "data"), exist_ok=True)  


# ****注意在写入的过程中，他是以逗号分隔，不要用空格
with open(data_file, 'w') as f:
    f.write('NumRooms, Alley, Price\n')
    f.write('NA,Pave, 127500\n')
    f.write('2, NA,10600\n')
    f.write('4, NA,178100\n')
    f.write('NA,NA,140000\n')
```

<span style = "color : blue">***以上介绍了csv的用法，（更详细的用法请见豆包）***</span>

```python
data = pd.read_csv(data_file)
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]

# inputs = inputs.fillna(inputs.mean()) # 导致 Pandas 在计算 .mean() 的时候碰到非数值数据报错。

# Fill only numeric columns with their mean
num_cols = inputs.select_dtypes(include='number').columns
inputs[num_cols] = inputs[num_cols].fillna(inputs[num_cols].mean())
# .mean()用来计算平均值的方法
# 通过把有参的数字计算完平均值后，就将那些无参的（NA）替换成平均值


print(inputs)
```



# 关于csv文件严格读取问题

### jupyter

![image-20250521215539772](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20250521215539772.png)

### vscode

![image-20250521215607950](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20250521215607950.png)



![image-20250521215449689](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20250521215449689.png)

> 注意：即使在.py文件中读取出了问题也可以正常打印，但是在jupyter中要求的比较严格

# 矩阵的转置

