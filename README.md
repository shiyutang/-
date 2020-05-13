[zhong](###重-要-的-问-题)

# Hands on deep learing 动手学深度学习笔记
每节课分为两部分，一部分记录部分知识点和探索的问题，另一部分记录代码和代码相关的笔记
每节课都会有代码，但是只有部分课会有笔记

代码来自于伯禹教育平台的[ElitesAI·动手学深度学习PyTorch版](https://www.boyuai.com/elites/course/cZu18YmweLv10OeV)

笔记总结于课程中的思考和课程讨论区

------------------------------

# 线性回归

* 定义：基于特征和标签之间的线性函数关系约束，线性回归通过建立单层神经网络，将神经网络中每一个神经元当成是函数关系中的一个参数，通过利用初始输出和目标输出建立损失，并优化损失最小的方式使得神经元的数值和真实函数参数数值最相近，从而通过网络训练得到最符合数据分布的函数关系。

* 实施步骤：
1. 初始通过随机化线性函数的参数，通过输入的x，会得到一系列y_h
2. 输出的y_h和真实值y之间因为神经元参数不正确产生差距，为了y_h和y能尽量地逼近，我们通过平方误差损失函数（MSE Loss）来描述这种误差。
3. 类似于通过求导得到损失函数最优解的方式，我们通过梯度下降法将这种误差传递到参数，通过调整参数使误差达到最小
4. 通过几轮的训练，我们得到的最小的损失值对应的神经元数值，就是描述输入输出的线性关系的最好的参数。

* 要点：
1. 确定输入输出之间一定满足线性关系，这一点可以通过对x,y画图看出，只有线性关系才能使用线性回归训练得到
2. 由于线性关系唯一由神经元个数决定，不同的参数个数唯一决定了这种线性关系，因此需要选择适合的特征用于线性回归

## 这一节中出现的有用的函数
1. 使用plt绘制散点图

```python
from matplotlib import pyplot as plt
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
```
2. 自行制作dataLoader: dataloader 为输入dataset可以随机获取dataset中batch size 个样本
> 通过使用打乱的indices，每次yield batch size个样本，生成的生成器可以用for调用
```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # random read 10 samples
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        # the last time may be not enough for a whole batch
        yield  features.index_select(0, j), labels.index_select(0, j)
```
3. 参数初始化：自行初始化网络中的参数，使用init模块
```python
from torch.nn import init

init.normal_(net[0].weight, mean=0.0, std=0.01)
init.constant_(net[0].bias, val=0.0)
```


## 重要的问题：
### 1.构建一个深度学习网络并训练需要哪些步骤？

深度学习网络的主要组成部分就是数据，网络和训练，因此可以根据这三部分展开为下面几个步骤：

#### 0. 数据部分
1. 生成数据集/找到现有数据集
2. 根据数据集构建Dataset 并用之构建dataloader
3. （可选）调用构建的Dataloader，得到数据并可视化，检查实现的正确性，并对数据有一定了解

#### 1. 网络部分
4. 定义模型，初始化模型参数
5. 定义损失函数，如本节的MSE loss
6. 定义优化函数，SGD,Adam... 及其参数：学习率，动量，weight_decay...

#### 2. 训练部分
7. 使用循环的方式，每个循环训练一遍所有数据
8. 将数据输入网络，根据损失函数和网络输出建立损失
9. 梯度清零，损失回传，优化器更新损失
10. 记录损失，可视化结果，往复训练

### 2.什么时候该用parameter.data?
下面是课程中使用的优化器的代码，可以发现，参数的更新使用了param.data
```python
def sgd(params, lr, batch_size): 
    for param in params:
        param.data -= lr * param.grad / batch_size # ues .data to operate param without gradient track
```
根据我的理解，这是由于反向传播机制在需要更新的参数进行运算时会构建动态运算图，如果直接使用这个`param`进行更新，就会在动态图中计入这一部分，从而反向传播时也会将这一步运算的梯度加入。而我们实际希望的则是损失函数对参数进行求导，而不希望再此参数上“节外生枝”。因此，在网络前向传播和损失函数计算之外的参数运算，应当使用param.data进行更新

-------------------------

# softmax和分类模型

## softmax
_定义_：softmax是用于将向量输出统一为有概率性质的一个函数，基于其指数的性质，它可以拉大大数和小数之间的差距，并且越大的数，为了达到和较小的数的差距只需要更小的差距，有助于筛选出一组向量中的最大值，并使其和其他数更加明显区分。

_性质_: softmax函数的常数不变性，即 softmax(x)=softmax(x+c)：对需要进行softmax变换的向量，同时减去一个较大的数不会对结果产生影响，推导如下：


<div align=center>  

![alt text](https://github.com/shiyutang/Hands-on-deep-learning/blob/master/%E5%9B%BE%E7%89%87/QQ%E5%9B%BE%E7%89%8720200214130612.png "softmax 常数不变性")   

</div>


优势：1. 通过softmax，输入向量每个元素的大小一目了然，不需要关心输出的具体数值，也不会存在输出相差较大时，不好比较的问题
2. 使用softmax的输出进行交叉熵函数的比较相当于统一了量纲，平衡了不同输出结果的重要程度


## 分类模型
_定义_：分类模型用于将输入的图片/信号进行分类，本单元讲到的分类模型是对输入进行一次线性变换，最后将分类结果利用softmax进行区分，示例如下图：


<div align=center> 

![Image Name](https://cdn.kesci.com/upload/image/q5hmymezog.png) 

</div>

_交叉熵函数_：上节的回归问题，我们用到了MSE，因为我们希望线性模型的参数和真实的完全一致，但是在分类模型中，我们只希望真实类别对应的概率越大越好，其他概率的具体数值对损失并没有过多含义。因此，这里使用了交叉熵函数，通过将标签化为one-hot编码，使得损失函数只关注正确类别对应的概率大小,其函数表达和实例操作如下：
<div align=center> 

![Image Name](https://github.com/shiyutang/Hands-on-deep-learning/blob/master/%E5%9B%BE%E7%89%87/%E4%BA%A4%E5%8F%89%E7%86%B5.png) 

</div>

如果标签为one-hot编码且只有一个正确类别，那么最小化交叉熵损失就是最大化正确类别对应输出概率的log形式，从另一个角度来看，最小化交叉熵损失函数等价于最大化训练数据集所有标签类别的联合预测概率

----------------------------

# 多层感知机
_定义_：多层感知机是采用多个神经网络层，并在其中添加了非线性单元，从而使得网络可以得到从输入到输出的非线性映射，从而将线性模型转化为非线性模型，在此基础上，增加网络的深度可以进一步增加网络的复杂度。

## 激活函数：
_定义_：拥有非线性部分的函数，即函数的导数并非处处一致，常见的激活函数有Relu，Leaky-relu，tanh，sigmoid等

实例：


<div align=center>  

![alt text](https://github.com/shiyutang/Hands-on-deep-learning/blob/master/%E5%9B%BE%E7%89%87/%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0.png "softmax 常数不变性")   

</div>


## 重要的问题：
### 1. 不同的激活函数应当如何选择？

* Relu：由于其计算效率和不容易梯度消失，从而大多数情况下使用；但是，由于其输出范围不确定，因此只能在隐藏层中使用；同时Relu函数因为在小于0的部分为0，因此容易在学习率大时使神经元失活

* Sigmoid函数：用于分类器时，通常效果更好，但是存在由于梯度消失。

* Tanh 函数: 和sigmoid 相似，但是其导数输出在（0，1），相比sigmoid更容易激活神经元；也存在梯度消失的问题



