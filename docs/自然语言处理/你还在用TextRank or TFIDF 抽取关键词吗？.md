# 什么！！！你还在用TextRank or TFIDF 抽取关键词吗？

> 本文着眼于简略地介绍关键词提取技术的前世今生



## 回顾历史

- 无监督
  - 统计模型
    - FirstPhrases
    - TfIdf
    - KPMiner [(El-Beltagy and Rafea, 2010)](http://www.aclweb.org/anthology/S10-1041.pdf)
    - YAKE [(Campos et al., 2020)](https://doi.org/10.1016/j.ins.2019.09.013)
  - 图模型
    - TextRank [(Mihalcea and Tarau, 2004)](http://www.aclweb.org/anthology/W04-3252.pdf)
    - SingleRank [(Wan and Xiao, 2008)](http://www.aclweb.org/anthology/C08-1122.pdf)
    - TopicRank [(Bougouin et al., 2013)](http://aclweb.org/anthology/I13-1062.pdf)
    - TopicalPageRank [(Sterckx et al., 2015)](http://users.intec.ugent.be/cdvelder/papers/2015/sterckx2015wwwb.pdf)
    - PositionRank [(Florescu and Caragea, 2017)](http://www.aclweb.org/anthology/P17-1102.pdf)
    - MultipartiteRank [(Boudin, 2018)](https://arxiv.org/abs/1803.08721)
- 有监督
  - Kea [(Witten et al., 2005)](https://www.cs.waikato.ac.nz/ml/publications/2005/chap_Witten-et-al_Windows.pdf)



### TFIDF

''
$$
\frac {  { count } ( w ) } { | D _ { i } | } \cdot \log \frac { N } { 1 + \sum _ { i = 1 } ^ { N } I ( w , D _ { i } ) }
$$



名字以及包含含义即即  【**词频**】  乘以  【**文档频率的倒数**】（取对数）

词频等于  【**该词出现次数**】  除以 【**本篇文章词总数**】

文档频率  等于  【**该词出现在多少文章中**】  除以 【**文章总数**】

(1为了防止分母为0)



### TextRank


$$
S(V_{i})=1-d+d \cdot \sum _{j \in In(v_{i})}\frac{1}{|out(v_{j})|}S(V_{j})
$$



在TextRank提取关键词算法中，限定窗口大小，构建词语共现网络，此时可构建无权无向图，也可根据出现次序构建无权有向图，根据PageRank算法迭代算出权重。实验证明无权无向图效果更好。（d是阻尼因子，防止外链为0的点出现Dead Ends问题）


$$
W S ( V _ { i } ) = ( 1 - d ) + d * \sum _ { j = 1 n ( V _ { i } ) } \sum _ { j = N _ { k } } ^ { W _ { i } } W _ { j i k } W S ( V _ { j } )
$$


而TextRank提取摘要算法中，构建的是有权无向图，节点是句子，权重是相似度，相似度的计算如下：


$$
 { s i m i l a r i t y } ( S _ { i } S _ { j } ) = \frac { | \left\{W_{k}\right\} |W_{k}\in S_{i}8W_{k}\in S_{j}| } { \log ( | S _ { i } | ) + \log ( | S _ { i } | ) }
$$


分子是两句子共同的词数量，分母是两个句子词数对数求和。



## 走进新时代



### bert也可以用来抽取关键词

![image-20220921010750371](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20220921010750371.png)



如图所示，将句子输入BERT模型，得到句子向量再与各个词的向量做余弦距离，得出关键词。


$$
\sin_{i}= \cos(w_{i},W)
$$


使用起来也非常简单：

```python
! pip install keybert
from keybert import KeyBERT
import jieba_fast
from tkitKeyBertBackend.TransformersBackend import TransformersBackend
from transformers import BertTokenizer, BertModel
doc = ""
seg_list = jieba_fast.cut(doc, cut_all=True)
doc = " ".join(seg_list)

tokenizer = BertTokenizer.from_pretrained('uer/chinese_roberta_L-2_H-128')
model = BertModel.from_pretrained("uer/chinese_roberta_L-2_H-128")

custom_embedder = TransformersBackend(embedding_model=model,tokenizer=tokenizer)
kw_model = KeyBERT(model=custom_embedder)
kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words=None)
```

![img](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/20220321123630313(1).png)



### 抽取关键词还可以预训练

![image-20220922222127755](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20220922222127755.png)



本篇论文最大的贡献在于提出了一种关键词生成的预训练手段，并且证明了此预训练方法对于其他下游任务如NER, QA, RE，summarization也有提升效果。

具体的，如上图所示，


$$
\mathcal{L}_{KBIR}(\theta)= \alpha \mathcal{L}_{MLM}(\theta)+ \gamma \mathcal{L}_{Infil}(\theta)+\sigma \mathcal{L}_{LP}(\theta)+ \delta \mathcal{L}_{KRC}(\theta)
$$




1. MLM：masked language model（masked Token Prediction） 单字符掩码任务
2. Infil：Keyphrase infilling 关键词填充任务，类似于 SpanBERT，掩码于整个关键词
3. LP：Length Prediction，关键词长度预测，将长度预测作为分类任务
4. KRC： Keyphrase Replacement Classification 随机替换同类关键词，二分类是否被替换了



**使用方法**

```python
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np

class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])

model_name = "ml6team/keyphrase-extraction-kbir-inspec"
extractor = KeyphraseExtractionPipeline(model=model_name)
```





## 引用

1. 统计学习方法第2版
1. Self-supervised Contextual Keyword and  Keyphrase Retrieval with Self-Labelling
1. Learning Rich Representation of Keyphrases from Text
1. PKE: an open source python-based keyphrase extraction toolkit.
1. Capturing Global Informativeness in Open Domain Keyphrase Extraction



<font size=24>关注本公众号，下期更精彩</font>

<img src="https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20220930221129484.png" alt="image-20220930221129484" style="zoom: 80%;" />

