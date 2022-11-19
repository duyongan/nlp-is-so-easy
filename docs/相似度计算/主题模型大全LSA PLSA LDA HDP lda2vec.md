# 主题模型大全LSA PLSA LDA HDP lda2vec



[TOC]



## 主题模型

所有主题模型都基于相同的假设：

1. 每个文档包含多个主题
2. 每个主题包含多个单词

<img src="https://notebook-media.oss-cn-beijing.aliyuncs.com/img/20190905111020331.png" style="zoom: 67%;" />

### LSA  

<img src="https://notebook-media.oss-cn-beijing.aliyuncs.com/img/v2-2044c66c00dcec7fa7042e6cf8e30187_r.jpg" style="zoom: 67%;" />

将文章X单词矩阵进行SVD分解，分解为文章（句子）X主题、主题X主题、主题X单词单个矩阵，其中文章（句子）X主题作为文章（句子）向量。



### PLSA   

![img](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/v2-0992d153c2cf9b1ed8ccb30228a15582_b.jpg)

![](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/v2-8d53e801e8dbada3c73bc9f4245df4ec_b.jpg)

1. *d*和 *w* 是已经观测到的变量，而 *z* 是未知的变量（主题），和LSA的矩阵分解是对应的。

2. 最大的矩形里装有*N*篇文档，每一篇文档 *d* 自身有个概率$P(D)$，从文档 *d* 到主题 *z* 概率分布$P(Z|D)$，随后从主题到词概率分布$P(W|Z)$，由此构成了 *w* 和 *z* 的联合概率分布$P(D,W)$，由此PLSA的参数量为 $zd+wz$ 。
3. 用EM算法求解模型



### LDA

PLSA假设分布为固定参数，容易产生过拟合，LDA在起基础上，加入狄利克雷分布，相当于加入先验知识。训练样本量足够大，pLSA的效果可以等同LDA。

![](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/bd7bca88def67e50635295ba65543bec_1440w.png)

> 狄利克雷分布：分布的分布，Beta分布的多元推广为狄利克雷分布，Beta分布是伯努利分布，二项分布的共轭先验（先验分布是beta分布，后验分也是beta分布）。
>
> Beta分布常用于AB测试当中，例如已知某页面点击率正常范围在0.2~0.35，点击率符合伯努利分布，点击率分布的分布符合Beta分布
>
> <img src="https://notebook-media.oss-cn-beijing.aliyuncs.com/img/v2-b813efe6c91c87474a11f877f9f6659a_720w.png" style="zoom: 80%;" />

1. α和β是dirichlet的参数，对于M篇文档都一样，用来控制每篇文档的两个概率分布
2. θ对于每一篇文档都一样，用来表征文档的主题分布
3. z是每个词的隐含主题，w是文档里的词
4. α和β是文集（corpus）级别的参数，所以在M篇文档外边
5. θ是文档级别的参数，在N个词盘子的外边
6. z和w是一一对应的，每个词对应一个隐含主题vec

$$
P(\theta , \vec{z}, \vec{w}|a, \beta)=P(\theta | \alpha)\prod _{n-1}^{N}P(z_{n}| \theta)P(w_{n}|z_{m}, \beta)
$$

下图大等边三角形内所有点到顶点对边距离和为1，代表生成三个单词的概率

小等边三角形内所有点到顶点对边距离代表生成三个主题的概率

小三角形中每个带叉的点都是一个PLSA模型，生成文档时，先根据叉点选择一个主题，再根据主题选择词

而LDA不再像PLSA是随机的点，而是小三角形中的曲线（一系列的点），即服从一定的分布，分布曲线由$\alpha \space \beta$ 决定

![](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/61eac4486ace521705156da208b285d9_1440w.webp)



### HDP-LDA

在LDA中，主题的个数是一个预先指定的超参数。通常可通过验证集和训练集得到最佳超参数。即当训练集评价指标下降，但验证集开始上升时，到达最优点。但有时最优个数会超大，此时可选择评价指标下降速度变慢的点。具体的评价指标（困惑度）如下：
$$
perplexity(D)=exp \left[ -\frac{\sum _{d=1}^{M}\log {p}(w_{i})}{\sum _{d=1}^{N}N_{d}}\right\}
$$
文档集合D，其中M为文档的总数， $w_d$ 为文档d中单词所组成的词袋向量，$ p(w_d)$ 为模型所预测的文档d的生成概率， $N_d$ 为文档d中单词的总数。

另外一种方法是在LDA基础之上融入**分层狄利克雷过程**（Hierarchical Dirichlet Process，HDP），构成一种非参数主题模型HDP-LDA。不需要提前预制主题个数，但模型参数求解更复杂。



**HDP构造过程**

类似于一个中国餐馆（一个文档），每个餐桌代表一个类别，从第一个顾客（一个单词）进来选桌子开始到最后一个结束，即可得出类别数。

n为已经选择好类别的单词的个数，$n_j$为第 *j* 个类别的单词数，下一个单词被分到已存在类别的概率： $\frac { n _ { j } } { n + \alpha }$，新类别的概率： $\frac{\alpha}{n+ \alpha}$。可知 $\alpha$ 越大，选择已存在类别概率越小，最终类别数越多；第 *j* 个类别的单词数越多，下个单词分到该类别可能性越大。



### lda2vec

其实就是：**word2vec +  lda**，lda的训练也变成了深度学习的方式，初始文档主题分布矩阵，在输出层文档主题的输出加上word2vec输出。

![](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/lda2vec_network_publish_text_header.gif)



## 引用

1. https://multithreaded.stitchfix.com/blog/2016/05/27/lda2vec/#topic=38&lambda=1&term=
1. https://jmlr.org/papers/volume3/blei03a/blei03a.pdf
1. 统计自然语言处理



<font size=24>关注本公众号，下期更精彩</font>

<img src="https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20220930221129484.png" alt="image-20220930221129484" style="zoom: 80%;" />