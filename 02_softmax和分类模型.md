## softmax和分类模型

### softmax
_定义_：softmax是用于将向量输出统一为有概率性质的一个函数，基于其指数的性质，它可以拉大大数和小数之间的差距，并且越大的数，为了达到和较小的数的差距只需要更小的差距，有助于筛选出一组向量中的最大值，并使其和其他数更加明显区分。

_性质_: softmax函数的常数不变性，即 softmax(x)=softmax(x+c)：对需要进行softmax变换的向量，同时减去一个较大的数不会对结果产生影响，推导如下：


<div align=center>  

![alt text](https://github.com/shiyutang/Hands-on-deep-learning/blob/master/%E5%9B%BE%E7%89%87/QQ%E5%9B%BE%E7%89%8720200214130612.png "softmax 常数不变性")   

</div>


优势：1. 通过softmax，输入向量每个元素的大小一目了然，不需要关心输出的具体数值，也不会存在输出相差较大时，不好比较的问题
2. 使用softmax的输出进行交叉熵函数的比较相当于统一了量纲，平衡了不同输出结果的重要程度


### 分类模型
_定义_：分类模型用于将输入的图片/信号进行分类，本单元讲到的分类模型是对输入进行一次线性变换，最后将分类结果利用softmax进行区分，示例如下图：


<div align=center> 

![Image Name](https://cdn.kesci.com/upload/image/q5hmymezog.png) 

</div>

_交叉熵函数_：上节的回归问题，我们用到了MSE，因为我们希望线性模型的参数和真实的完全一致，但是在分类模型中，我们只希望真实类别对应的概率越大越好，其他概率的具体数值对损失并没有过多含义。因此，这里使用了交叉熵函数，通过将标签化为one-hot编码，使得损失函数只关注正确类别对应的概率大小,其函数表达和实例操作如下：
<div align=center> 

![Image Name](https://github.com/shiyutang/Hands-on-deep-learning/blob/master/%E5%9B%BE%E7%89%87/%E4%BA%A4%E5%8F%89%E7%86%B5.png) 

</div>

如果标签为one-hot编码且只有一个正确类别，那么最小化交叉熵损失就是最大化正确类别对应输出概率的log形式，从另一个角度来看，最小化交叉熵损失函数等价于最大化训练数据集所有标签类别的联合预测概率




