# KBQA新方法

以往的KBQA步骤范式为：

1. 实体识别
2. 实体链接
3. 意图识别（or 关系or 属性识别）
4. 查询数据库

而本文中介绍的来自美团[^1]的方案是：

1. 实体识别
2. 实体链接
3. 查询数据库（控制步数）
4. 关系学习模型打分



## 引子

接下来将详细介绍论文中的方法，但在介绍之前，先看看2021年3月份一篇关于知识补全的文章BERTRL[^2]：

![image-20220912160338451](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20220912160338451.png)

与以往将知识图谱用Trans系列或图神经网络建模的方式不同，文章中将知识图谱线性化输入到Bert中。文章中提及了两种线性化图谱的方式：

1. 将所有涉及待预测量两个实体之间的路径同时输入模型

2. 分别将每种路径单独输入模型，之后分数聚合（文中的聚合方式为取分数最大值）
   $$
   { s c o r e } ( h , r , t ) =   { max } P ( y = 1 | h , r , t , h \frac { p } { t } , t )
   $$
   

这里举一个输入的例子：[CLS]姚明的妻子是谁？是叶莉吗？[SEP]姚明的女儿是姚沁蕾；姚沁蕾的妈妈是叶莉

如此一来，就融合了知识图谱的知识以及语义知识了。提到这，是不是想起被Jena知识推理工具支配的恐惧，Jena的知识推理是基于经验主义的规则系统，召回率极低，但准确率高，代码举例：

```shell
[rule: (?a :女儿 ?b)(?b :妈妈 ?c) -> (?a :夫妻 ?c)]
```

有关jena的工程化实践，有时间会细说。



## 主文

前文说过有两种线性化图谱的方式，BERTRL用了单独输入各个路径的方式，而美团这篇用了另一种，平铺一次性输入，举例：

[CLS]姚明的妻子是谁？是叶莉吗？[SEP] 姚明的女儿是姚沁蕾；姚沁蕾的妈妈是叶莉 [SEP] 姚明的母亲是方凤娣；方凤娣的儿媳是叶莉



### 微调方法

![image-20220912172548740](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20220912172548740.png)



将全部路径一股脑输入模型，预测档期那候选实体是否为答案，之后依次将所有候选实体算出分数，即达到了目的。



### 预训练方法

![image-20220912172622142](https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20220912172622142.png)



除了直接使用Bert模型进行进行微调之外，作者还提供了一种预训练的方法，预训练之后再进行微调可有效提高KBQA效果。作者提出了三种任务：

1. 关系抽取

   [CLS]句子[SEP]头实体h, 关系r, 尾实体t[SEP]  根据句子预测头实体h和尾实体t是否具有关系r

2. 关系匹配  

   [CLS]句子1 [SEP]句子2 [SEP]  预测句子1和句子2是否具有相同的关系

3. 关系推理即BERTRL



## 引用

1. Large-Scale Relation Learning for Question Answering over Knowledge Bases with Pre-trained Language Models
2. Inductive Relation Prediction by BERT



<font size=24>关注本公众号，下期更精彩</font>

<img src="https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20220930221129484.png" alt="image-20220930221129484" style="zoom: 80%;" />











