# 由词向量word2vec、fastext如何表征句子向量



[TOC]

## word2vec

<img src="https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20221004154743231.png" alt="image-20221004154743231" style="zoom: 28%;" />

#### 两种训练方式

训练的目标是单词预测，但输出单词向量的是隐藏层

1. CBOW：根据上下文预测当前值
2. Skip-gram：根据当前词预测上下文

**注意**：skip-gram的训练过程**不是一次性用中心词预测四个词**，而是中心词和**一个周围词**组成一个训练样本

<img src="https://notebook-media.oss-cn-beijing.aliyuncs.com/img/okaz1rhibn.png" alt="img" style="zoom: 50%;" />

#### 加快训练速度的两种方法



1. 采样率

$$
P(w_{i})=(\sqrt{\frac{z(w_{i})}{0.001}}+1)\cdot \frac{0.001}{z(w_{i})}
$$

​		$z(w_i)$：单词$w_i$出现的次数与总单词个数的比值

​		$P(w_i)$：是保留该单词的概率	

​		<img src="https://notebook-media.oss-cn-beijing.aliyuncs.com/img/5doyeeejf3.png" alt="img" style="zoom:67%;" />

​		由图可知单词$w_i$出现的次数越多，保留概率越低，避免重复无用的训练





2. 负采样率

$$
P(w_{i})= \frac{f(w_{i})^{3/4}}{\sum _{j=0}^{n}(f(w_{j})^{3/4})}
$$
<img src="https://notebook-media.oss-cn-beijing.aliyuncs.com/img/ppqqy0hpe3.png" alt="img" style="zoom: 67%;" />

​		$f(w_i)$：词频

​		词频越高被负采样概率越高。





3. 层次softmax

​		Huffman Tree （最优二叉树：最重要的放在最前面）

<img src="https://notebook-media.oss-cn-beijing.aliyuncs.com/img/v2-19a625c2e625bf785545861011477a6b_r.jpg" style="zoom: 33%;" />

​		图b为最优二叉树，带权路径长度计算：

​		图a：$WPL = 5 * 2 + 7 * 2 + 2 * 2 +13 * 2 = 54$

​		图b：$WPL = 5 * 3 + 2 * 3 + 7 * 2 + 13 * 1 = 48$

​		最优二叉树的构造过程如下图：

<img src="https://notebook-media.oss-cn-beijing.aliyuncs.com/img/v2-b88babe61e4b10ab66fd7efc83208853_r.jpg" style="zoom: 50%;" />

​	编码方法：左子树为0，右子树为1

​	所以D编码为0，B编码为10，C编码为110，A编码为111

​	那么输出层的训练目标可以从词典数量缩减到最优二叉树的深度了，用$sigmoid$二分类预测每一位编码。





优点：使用向量时直接KV取值，速度快

缺点：

​			1. 无法解决一词多义问题

​			2. OOV：超出词典无法表征，没有词序====>解决之道【FastText】

​			3. 句子内n-grams，天生具有词序信息   

> “我 爱 她”如果加入 2-Ngram，加入特征 “我-爱” 和 “爱-她”，“我 爱 她” 和 “她 爱 我” 就能区别开了。
>
> Hash解决n-gram膨胀问题，如n-gram数为10240，设置n-gram最大词典数为1024，取余再转向量
>
> Hash冲突问题实际对效果影响不大

​			4. 词内n-grams，即subword，解决未登录词的问题。



### 1. 平均

句子$S$中所有词向量加起来求平均
$$
V=\frac{\sum_{w_i \in S} v_{w_i}}{N}
$$

### 2. 加权平均

TFIDF作为权重，对词向量加权求平均
$$
V=\frac{\sum_{w_i \in S} (TF \cdot IDF) v_{w_i}}{N}
$$

$$
TF \cdot IDF = \frac{count(w)}{|D_{i}|} \cdot \log \frac{N}{1+ \sum _{i=1}^{N}I(w,D_{i})}
$$

1. $count(w)$：文档$D_i$中词$w$的数量。
2. $|D_i|$：文档$D_i$中所有词的数量。
3. $N$：文档总数。
4. $I(w,Di)$：文档Di是否包含关键词，若包含则为1，若不包含则为0。



### 3. SIF

加权平均换为平滑逆词频，a为调节参数
$$
w=\frac{a}{a+TF}
$$
 加权平均之后再减去句子矩阵经过SVD的主成分

![image-20221008000824233](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20221008000824233.png) 

**SVD图示**

![img](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/v2-4437f7678e8479bbc37fd965839259d2_r.png)

**代码理解**

```python
from sklearn.decomposition import TruncatedSVD
X=np.random.random((20,128))
svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
svd.fit(X)
pc = svd.components_
X = X - X.dot(pc.transpose()) * pc
pc.transpose().shape #(128, 1)
X.dot(pc.transpose()).shape  #(20, 1)
pc.shape #(1, 128)
```



### 4. doc2vec

doc2vec与word2vec一样也有两种训练方式CBOW和Skip-gram。Doc2vec与word2vec的不同在于，在输入层增加了一个句子向量Paragraph vector，在同一个句子的若干次训练中是共享的，可以看作句子的主旨。

<img src="https://notebook-media.oss-cn-beijing.aliyuncs.com/img/v2-96b17c2315ce71c070acf6e6e0f9b28f_r.jpg" style="zoom:50%;" />

<img src="https://notebook-media.oss-cn-beijing.aliyuncs.com/img/v2-74929b5858f4a5c26cfaa496d494ad5b_b.jpg" alt="v2-74929b5858f4a5c26cfaa496d494ad5b_b" style="zoom:50%;" />





## 引用

1. Efficient estimation of word representations in vector space
1. 统计自然语言处理







<font size=24>关注本公众号，下期更精彩</font>

<img src="https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20220930221129484.png" alt="image-20220930221129484" style="zoom: 80%;" />





