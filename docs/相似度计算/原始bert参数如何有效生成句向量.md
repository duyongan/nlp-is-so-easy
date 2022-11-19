# 原始bert参数如何有效生成句向量



**目录**

[TOC]



## BERT

![image-20221010141651121](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20221010141651121.png)

### CLS

直接使用[CLS]位置向量作为句向量                                                   



### last average

最后一层词向量（即除[CLS]之外）求平均



### first last average

第一层和最后一层词向量平均



### SBERT

![image-20221012110535823](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20221012110535823.png)

sbert采用经典的双塔方式，提出了三种目标函数：

1. 拼接两个句子向量，直接二分类（Classification Objective Function）
   $$
   softmax(W_{t}(u,v,|u-v|))
   $$
   选择$(u,v,|u-v|)$拼接，mean pooling方式效果最好

   ![image-20221012155027719](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20221012155027719.png)

2. 利用两个句子的余弦距离的mean-squared-error作为损失函数（Regression Objective Function）

   以下为论文中的实现

   

   ```python
   class CosineSimilarityLoss(nn.Module):
       def __init__(self, model: SentenceTransformer, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity()):
           super(CosineSimilarityLoss, self).__init__()
           self.model = model
           self.loss_fct = loss_fct
           self.cos_score_transformation = cos_score_transformation
   
   
       def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
           embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
           output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
           return self.loss_fct(output, labels.view(-1))
   ```

   

3. 每次给一个正例，一个负例，增大负例距离，减小正例距离（Triplet Objective Function）。$s_x$为句子向量$||\cdot||$是一种距离衡量，间隔 $\epsilon$ 保证正例$s_p$(positive) 比负例 $s_n$(negative) 至少要近$\epsilon$。

$$
\max(||s_{a}-s_{p}||-||s_{a}-s_{n}||+ \epsilon ,0)
$$



#### 在STS数据的测评

##### 无监督

在无监督领域实验表明无论是bert-cls还是bert-avg效果都还不如直接Glove

> GloVe与word2vec的区别
>
> word2vec：根据上下文呢预测中间的词汇，或者根据中间的词汇预测上下文。
>
> GloVe：构建词汇的共现矩阵，再进行类似PCA的操作

![image-20221012162242596](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20221012162242596.png)

##### 有监督

作者使用Regression Objective Function损失函数的方式进行训练得出以下测评：

![image-20221012164309419](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20221012164309419.png)



### Bert-flow

从上文中我们知道，bert在没有fine-turn的情况下，无论使用CLS还是avg效果都很差。为啥呢？

1. CLS预训练的目标是判断两个句子是否存在延续性，【连续性和相似大相径庭】。
2. 各向异性：各个维度衡量标准不一样(左图为各向异性，右图为各向同性)

<img src="https://notebook-media.oss-cn-beijing.aliyuncs.com/img/v2-287a4e8478b38ac7dd401c503ddb97c0_720w.png" style="zoom: 67%;" />

​		由下图可知word2vec各向分布较为一致，而transformer差异性很大。

![image-20221012173359353](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20221012173359353.png)

​		各向异性会导致向量堆积在一起，余弦相似度都很高

![](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/v2-16afeb2bcb31948dded07183b983b609_720w.jpg)

3. bert 空间向量所属的坐标系并非标准正交基。
   $$
   \cos(x,y)= \frac{\sum _{i=1}^{d}x_{i}y_{i}}{\sqrt{\sum _{i=1}^{d}x_{i}}\sqrt{\sum _{i=1}^{d}y_{i}}}
   $$
   余弦相似度计算方式是在标准正交基下成立的，如果如果符合标准正交基，语义学习的好，效果应该是好的。

   > 标准正交基：一个内积空间的正交基（orthogonal basis）是元素两两正交的基。称基中的元素为基向量。假若，一个正交基的基向量的模长都是单位长度1，则称这正交基为标准正交基



如何解决呢？

bert-flow提出的解决方案是从各向异性入手的：将原始的bert句子向量空间转换到标准高斯分布空间。

![image-20221012174752657](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20221012174752657.png)

$u$是bert空间向量（observe space），$z$是高斯空间向量（latent space），服从$p_z(z)$的先验分布。$f$是通过【**训练**】（固定bert参数，只迭代预训练转化函数参数）得到的从高斯空间转换到bert空间的可逆函数。
$$
z \sim p_z(z),u=f_{\phi}(z)
$$

根据流模型（flow model）原理，bert空间转换为高斯空间如下（$det$：行列式；$f^{-1}$：$f$ 的逆函数）：
$$
p_u(u)=p_z(f_{\phi}^{-1}(u))|det \frac{\partial f_{\phi}^{-1}(u)}{\partial u}|
$$
最大化对数似然函数（log likelihood）
$$
\max_{\phi} \mathbb E_u=BERT(sentence),sentence \sim \mathcal D
$$

$$
\log p_z(f_{\phi}^{-1}(u))+ \log |det \frac{\partial f_{\phi}^{-1}(u)}{\partial u}|
$$

#### 流模型（flow model）

![Snipaste_2022-10-13_11-53-22](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-10-13_11-53-22.png)

定义原始数据分布为简单的高斯分布$\pi(z)$，且$𝑥=G(z)$ 。转换之后的x服从$p_G(x)$分布。此时的目标函数最大化对数似然等于最小化KL散度。

 **一些基础知识**

 1. 雅可比矩阵（Jacobian）

    定义$𝑥=f(z)$ ,$f$为可逆函数即$z=f^{-1}(x)$。如果设x和z都是二维的，那么此时Jacobian matrix（以下简称$J_f$）如下图所示。

    所以如果$z=[z_1,z_2],x=[z_1+z_2,2z_2]$，根据$J_{f}$的求偏导的规则可求得$J_{f}$ ，同理可得$J_{f^{-1}}$，同时可知$J_{f}J_{f^{-1}}=I$。

![image-20221013151032918](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20221013151032918.png)

​    

 2. 行列式（Determinant）

    计算方式：$$D= \sum(-1)^{k}a_{1k_{1}}a_{2k_{2}}\cdots a_{nk}$$

    

    

![image-20221013151354679](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20221013151354679.png)

​    

​    

行列式的几何意义其实是高维空间的“体积”。


​    

​    

![image-20221013151312730](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20221013151312730.png)

​    

 3. 随机变量的变量替换定理(Change of Variable Theorem)

    定义z服从$\pi(z)$，x服从$p(x)$分布，$𝑥=f(z)$ 。x可以z通过f得到，那么$\pi(z)$和$p(x)$有什么关系呢？

    

![image-20221013151446650](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20221013151446650.png)

​    

假设$\pi(z)$和$p(x)$都服从最简单的均匀分布（Uniform Distribution），$\pi(z)\sim U(0,1) \ ,\  p(x)\sim U(1,3)$。为了保证积分面积都为1，则$\pi(z)=1\ ,\ p(x)=0.5$，即$p(x^{\prime})= \frac{1}{2}\pi(z^{\prime})$。


​    

![image-20221013151609476](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20221013151609476.png)

​    

更一般的情况，复杂的分布可以取其中一小段视为均匀分布，由此可以得到$p(x^{\prime})= \pi(z^{\prime})| \frac{dz}{dx}|$。（有可能反向对应，要加绝对值）


​    

![image-20221013151641573](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20221013151641573.png)

​    

​    

推广到二维的情况（别忘了，det的含义是面积）：



![image-20221013151844217](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20221013151844217.png)

​    

经过以下的推到可以得到两个分布之间的推导公式：


​    

![image-20221013151937288](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20221013151937288.png)

​    

所以将似然函数中$P_G$替换成分布之间的推导公式，将$z^i$替换为向量之间的推导公式，可得到最终的目标函数。


​    

![image-20221013152047399](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20221013152047399.png)







### Bert-whitening

bert-flow为了能够转化分布，仍然需要训练一个转换参数。Bert-whitening采用了更加直接的办法，不需要再去迭代预训练，从标准正交基入手，利用白化的手段将bert向量转换到以标准正交基为基底的向量。

![image-20221014010817361](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20221014010817361.png)

经过白化操作的数据具有以下特征：

- 消除了特征之间的相关性（协方差矩阵除对角线都为0）
- 所有特征的方差都为 1（协方差矩阵对角线都为1）

> PCA白化：$W= \Lambda ^{-1/2}U^{T}$
>
> ZCA白化：$W=U \Lambda ^{-1/2}U^{T}$
>
> 同：都消除了特征之间相关性，同时使得所有特征的方差都为 1。
>
> 异：相比于PCA白化，ZCA白化把数据旋转回了原来的特征空间，处理后的数据更接近原始数据。

设向量集合为$\left\{x_{i}\right\} _{i=1}^{N}$，标准的高斯分布均值为0，协方差为单位阵这与白化操作不谋而合。想要均值为0直接$x_i-\mu $，其中$\mu= \frac{1}{N}\sum _{i=1}^{N}x_{i}$ 。接下来将$x_i-\mu $ 执行线性变化$(x_{i}- \mu)W$，使得转换后的数据协方差矩阵为单位阵。
$$
\tilde{x}_{i}=(x_{i}- \mu)W
$$

在白化操作中，$W$给出了解法：

矩阵$X$白化至矩阵$Y$，根据协方差求解公式，$X$协方差$D_X=\frac{1}{m}XX^{T}$，所以：
$$
D_Y= \frac{1}{m}YY^{T}= \frac{1}{m}(WX)(WX)^{T}
$$

$$
= \frac{1}{m}WXX^{T}W^{T}=W(\frac{1}{m}XX^{T})W^{T}=WD_XW^{T}
$$

白化的目标是使得$\tilde{\Sigma}=W^{T}\Sigma W=I$，所以可以推导出：
$$
 \Rightarrow \Sigma =(W^{T})^{-1}W^{-1}=(W^{-1})^{T}W^{-1}
$$
$\Sigma$是一个半正定对称矩阵，数据够多时通常是正定的，从而$\Sigma=U \Lambda U^{T}$，$U$是正交矩阵，而$\Lambda$是对角阵，并且对角线元素都是正的，那么可以令$W^{-1}= \sqrt{\Lambda}U^{T}$,即$W=U \sqrt{\Lambda ^{-1}}$。

Bert-whitening使用的是PCA白化操作，其实可以进一步再承上$U$，完成ZCA白化操作，转换至原始特征空间。

Bert-whitening即不需叠加训练的方式，也达到了和flow相同的水准，还降低了维度，可谓一举三得。


![image-20221013175410120](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20221013175410120.png)





## 带参Bert-whitening

上文提到白化操作最终推导公式为：
$$
\tilde{x}_{i}=(x_{i}- \mu)U \sqrt{\Lambda ^{-1}}
$$
从评测中可知有些数据集评测指标相比first-last-avg是下降的，还不如不标准正交。解决之道：增加两个超参数，控制转换力度。
$$
\tilde{x}_{i}=(x_{i}- \beta \mu)U \Lambda ^{- \gamma /2}
$$
这样总有一种力度能达到最佳。



## CoSent

Sbert存在的问题是训练和预测的不一致，而如果直接优化预测目标cos，效果往往很差。解决之道：利用排序学习的方式，使得正样本对的距离都小于负样本对的距离。
$$
\log(1+ \sum _{\sin(i,j)> \sin(k,l)}e^{\lambda(\cos(u_{k},u_l)- \cos(u_{i},u_{j}))})
$$


<img src="https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20221021163231894.png" alt="image-20221021163231894" style="zoom: 33%;" />



```python
def cosent_loss(y_true, y_pred):
    """排序交叉熵
    y_true：标签/打分，y_pred：句向量
    """
    y_true = y_true[::2, 0]
    y_true = K.cast(y_true[:, None] < y_true[None, :], K.floatx())
    y_pred = K.l2_normalize(y_pred, axis=1)
    y_pred = K.sum(y_pred[::2] * y_pred[1::2], axis=1) * 20
    y_pred = y_pred[:, None] - y_pred[None, :]  #cos(u_{k},u_l)- cos(u_{i},u_{j})
    y_pred = K.reshape(y_pred - (1 - y_true) * 1e12, [-1])  #筛选sin(i,j)> sin(k,l)
    y_pred = K.concatenate([[0], y_pred], axis=0)  # e^0=1
    return K.logsumexp(y_pred)
```







## 引用

1. （2019）BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
1. （2019）Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
1. （2019）Improving Neural Language Generation With Spectrum Control
1. （2019）Representation Degeneration Problem In Training Natural Language Generation Models
1. （2020）On the Sentence Embeddings from Pre-trained Language Models
1. Flow-based  Generative Model 
1. （2021）Whitening Sentence Representations for Better Semantics and Faster Retrieval
1. （2022.5）带参bert-whitening：https://spaces.ac.cn/archives/9079
1. （2022.1）CoSENT：https://kexue.fm/archives/8847





<font size=24>关注本公众号，下期更精彩</font>

<img src="https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20220930221129484.png" alt="image-20220930221129484" style="zoom: 80%;" />





