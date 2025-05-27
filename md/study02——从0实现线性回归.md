# 实现线性回归

## torch.normal

`torch.normal()` 是 PyTorch 里用于从正态分布（也叫高斯分布）中抽样生成随机数的函数。下面为你介绍它的用法、参数以及示例。

### 函数定义

```python
torch.normal(mean, std, *, generator=None, out=None)
```

### 参数说明

- **mean**：该参数代表分布的均值，可以是标量，也能是张量。
- **std**：此参数表示分布的标准差，同样可以是标量或者张量。
- **generator**（可选）：若要使用自定义的随机数生成器，可通过该参数指定。
- **out**（可选）：用于存储输出结果的张量。

### 使用示例

#### 1. 标量均值与标量标准差

```python
import torch

# 生成一个从 N(mean=0, std=1) 分布中抽样的随机数
random_number = torch.normal(mean=0.0, std=1.0)
print(random_number)  # 输出类似 tensor(-0.4567)

# 生成一个 3x3 大小的张量，其元素从 N(mean=2, std=0.5) 分布中抽样
random_tensor = torch.normal(mean=2.0, std=0.5, size=(3, 3))
print(random_tensor)
```

#### 2. 张量均值与标量标准差

```python
# 生成一个均值张量
mean_tensor = torch.tensor([1.0, 2.0, 3.0])

# 为每个均值生成一个对应的随机数，标准差固定为 0.1
random_tensor = torch.normal(mean=mean_tensor, std=0.1)
print(random_tensor)  # 输出类似 tensor([1.0521, 2.0345, 2.9876])
```

#### 3. 张量均值与张量标准差

```python
# 均值张量和标准差张量的形状需保持一致
mean_tensor = torch.tensor([1.0, 2.0, 3.0])
std_tensor = torch.tensor([0.1, 0.2, 0.3])

# 从不同的正态分布中抽样
random_tensor = torch.normal(mean=mean_tensor, std=std_tensor)
print(random_tensor)  # 输出类似 tensor([1.0876, 1.8923, 3.2765])
```

### 注意要点

- 当 `mean` 和 `std` 都为标量时，必须通过 `size` 参数来指定输出张量的形状。

- 当 `mean` 或 `std` 为张量时，输出张量的形状会与输入张量的形状相同。

- 正态分布的概率密度函数公式为：*f*(*x*)=*σ*2*π*1*e*−2*σ*2(*x*−*μ*)2其中，*μ* 是均值，*σ* 是标准差。
## matmul

在编程领域，尤其是深度学习和科学计算中，`matmul` 通常指矩阵乘法操作。矩阵乘法是线性代数的核心运算，在不同的编程语言和库中有不同的实现方式。以下是一些常见的实现及其用法：

### 1. **Python NumPy 中的 `np.matmul`**

NumPy 是 Python 中用于科学计算的基础库，提供了 `matmul` 函数用于矩阵乘法。

```python
import numpy as np

# 示例：二维矩阵乘法
A = np.array([[1, 2], [3, 4]])  # 形状：(2, 2)
B = np.array([[5, 6], [7, 8]])  # 形状：(2, 2)
C = np.matmul(A, B)
# 输出：[[19 22], [43 50]]

# 示例：矩阵与向量乘法
A = np.array([[1, 2], [3, 4]])  # 形状：(2, 2)
b = np.array([5, 6])            # 形状：(2,)
c = np.matmul(A, b)
# 输出：[17 39]
```

**注意事项**：

- 两个矩阵维度必须满足：`A.shape[-1] == B.shape[-2]`。
- 支持广播（Broadcast）机制，可处理高维数组。

### 2. **PyTorch 中的 `torch.matmul` 或 `@` 运算符**

PyTorch 是深度学习框架，`matmul` 用于张量乘法，与 NumPy 类似但支持 GPU 加速。

```python
import torch

# 示例：二维矩阵乘法
A = torch.tensor([[1, 2], [3, 4]])  # 形状：(2, 2)
B = torch.tensor([[5, 6], [7, 8]])  # 形状：(2, 2)
C = torch.matmul(A, B)  # 或 C = A @ B
# 输出：tensor([[19, 22], [43, 50]])

# 示例：批量矩阵乘法（处理多个矩阵）
A = torch.randn(3, 2, 4)  # 3个 (2, 4) 矩阵
B = torch.randn(3, 4, 5)  # 3个 (4, 5) 矩阵
C = torch.matmul(A, B)    # 输出：3个 (2, 5) 矩阵
```

**注意事项**：

- 高维张量乘法会将前导维度视为批量维度，逐批进行矩阵乘法。

### 3. **TensorFlow 中的 `tf.matmul`**

TensorFlow 是另一个深度学习框架，提供 `matmul` 函数用于张量乘法。

```python
import tensorflow as tf

# 示例：二维矩阵乘法
A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.matmul(A, B)
# 输出：tf.Tensor([[19 22], [43 50]], shape=(2, 2), dtype=int32)
```

**注意事项**：

- 与 PyTorch 类似，支持高维张量和 GPU 加速。

### 4. **数学定义**

矩阵乘法的数学定义：
若 *A* 是 *m*×*n* 矩阵，*B* 是 *n*×*p* 矩阵，则 *C*=*A*×*B* 是 *m*×*p* 矩阵，其中：
$$
C_{i,j}=\sum_{k = 1}^{n} A_{i,k}\times B_{k,j}
$$


### 5. **常见错误**

- **维度不匹配**：确保 `A` 的列数等于 `B` 的行数。
- 混淆 `matmul` 与 `*`：
  - `matmul`：矩阵乘法。
  - `*` 或 `multiply`：逐元素乘法（点乘）。

### 6. **性能优化**

矩阵乘法是计算密集型操作，现代库（如 PyTorch、TensorFlow）会利用 GPU/TPU 加速。例如：

```python
# PyTorch GPU 加速示例
A = torch.randn(1000, 1000).cuda()
B = torch.randn(1000, 1000).cuda()
C = torch.matmul(A, B)  # 在 GPU 上执行
```

如果需要针对特定场景（如稀疏矩阵、批处理）的更详细实现，可以进一步说明！

## matplotlib

官方文档：https://matplotlib.org/stable/users/index.html

菜鸟教程：https://www.runoob.com/matplotlib/matplotlib-tutorial.html

## yield

在 Python 中，`yield` 是一个非常重要的关键字，主要用于定义**生成器函数**（Generator Function）。生成器是一种特殊的迭代器，它允许你在需要时逐个生成值，而不是一次性生成所有值，从而节省内存和提高效率。

### 1. 什么是生成器？

生成器是一种特殊的函数，使用 `yield` 而不是 `return` 来返回值。与普通函数不同，生成器在每次调用时不会从头开始执行，而是从上一次 `yield` 的位置继续执行。

#### 简单示例：

```python
def my_generator():
    yield 1
    yield 2
    yield 3

# 创建生成器对象
gen = my_generator()

# 使用 next() 函数获取生成器的下一个值
print(next(gen))  # 输出：1
print(next(gen))  # 输出：2
print(next(gen))  # 输出：3
print(next(gen))  # 报错：StopIteration（所有值已生成完毕）
```

### 2. 为什么需要生成器？

#### 2.1 **节省内存**

当处理大量数据时，普通函数需要一次性生成并存储所有数据，而生成器是**惰性计算**（Lazy Evaluation），只在需要时生成值。

**对比示例**：

```python
# 普通函数：返回包含 1 到 1000 的列表
def get_list():
    result = []
    for i in range(1, 1001):
        result.append(i)
    return result

# 生成器函数：逐个生成 1 到 1000 的值
def get_generator():
    for i in range(1, 1001):
        yield i

# 使用普通函数（一次性生成所有值）
my_list = get_list()
for num in my_list:
    pass  # 处理每个数

# 使用生成器（每次只生成一个值）
my_gen = get_generator()
for num in my_gen:
    pass  # 处理每个数
```

#### 2.2 **处理无限序列**

生成器可以表示无限序列，例如斐波那契数列：

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# 使用生成器获取前 10 个斐波那契数
gen = fibonacci()
for _ in range(10):
    print(next(gen))  # 输出：0, 1, 1, 2, 3, 5, 8, 13, 21, 34
```

### 3. `yield` 的工作原理

- **暂停和恢复**：当生成器遇到 `yield` 时，会暂停执行并返回当前值，下次调用时从暂停处继续执行。
- **状态保存**：生成器会保存局部变量和执行状态，直到下一次调用。

#### 示例解析：

```python
def count_up_to(n):
    count = 1
    while count <= n:
        yield count
        count += 1

# 创建生成器
counter = count_up_to(3)

# 第一次调用 next()
print(next(counter))  # 输出 1，暂停在 yield 处
# 第二次调用 next()
print(next(counter))  # 从 count += 1 继续，输出 2，再次暂停
# 第三次调用 next()
print(next(counter))  # 输出 3
# 第四次调用 next()
print(next(counter))  # 报错：StopIteration
```

### 4. 生成器的常见用法

#### 4.1 **遍历大文件**

处理大文件时，逐行读取而不是一次性加载整个文件：

```python
def read_large_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield line

# 使用生成器逐行处理文件
for line in read_large_file('huge_data.txt'):
    process(line)  # 处理每一行数据
```

#### 4.2 **生成器表达式**

类似于列表推导式，但使用圆括号 `()`：

```python
# 列表推导式：一次性生成所有值
squares_list = [x**2 for x in range(10)]  # [0, 1, 4, 9, ..., 81]

# 生成器表达式：惰性生成值
squares_gen = (x**2 for x in range(10))  # 生成器对象

# 逐个获取值
for num in squares_gen:
    print(num)
```

### 5. 生成器与迭代器的对比

| **特性**       | **生成器（Generator）**            | **迭代器（Iterator）**                 |
| -------------- | ---------------------------------- | -------------------------------------- |
| **定义方式**   | 使用 `yield` 关键字或生成器表达式  | 实现 `__iter__()` 和 `__next__()` 方法 |
| **状态保存**   | 自动保存局部变量和执行状态         | 需要手动管理状态变量                   |
| **代码简洁性** | 代码更简洁，无需显式实现迭代器协议 | 需要实现完整的迭代器协议               |
| **内存效率**   | 更高，惰性计算                     | 取决于具体实现                         |

### 6. 进阶：`yield from`（Python 3.3+）

`yield from` 用于简化嵌套生成器：

```python
def flatten_list(nested_list):
    for sublist in nested_list:
        yield from sublist  # 相当于 for item in sublist: yield item

# 使用示例
nested = [[1, 2], [3, 4], [5]]
for num in flatten_list(nested):
    print(num)  # 输出：1, 2, 3, 4, 5
```

### 总结

- **`yield` 的核心作用**：将函数转换为生成器，实现惰性计算，节省内存。
- **适用场景**：处理大数据集、无限序列、数据流等。
- **优点**：内存高效、代码简洁、可表示无限序列。

如果有具体的使用场景或问题，可以告诉我，我会进一步帮你解答！ 😊

## 细节问题

通过跟着敲完李沐的线性回归代码，让我感到对于一些细节性的问题不要出错，例如：多敲个字母导致不同类型的数据进行运算导致报错什么的！或者说在定义类型的时候粗心大意等等......这些问题也反映出了：我虽然说是能听懂，但是离掌握整体训练思路后自己将公式落实到代码里，自己去进行数据间的运算，亲手去架起这个思路，还是需要沉淀的！！！
