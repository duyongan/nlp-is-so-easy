# 没有比这更详细的推导  attention为什么除以根号dk——深入理解Bert系列文章



<font size=6>目录</font>

[TOC]

## 维持均值为0，方差为1的分布



### 基础知识

#### 随机变量

对随机事物的量化。例如硬币正反面是随机的，用0和1表示正反面，就成了随机数。

#### 期望

$$E(aX)=aE(X)$$

##### 连续型期望

$$E [ X ] = \int _ { - \infty } ^ { \infty } x f ( x ) d x$$

##### 离散型期望

$$E ( X ) = \sum _ { i = 1 } ^ { \infty } x _ { i } p _ { x_i }$$ 

##### 条件期望

随机变量X的条件期望$E(X|Y=y$) 依赖于Y的值y，即 $E(X|Y=y)$是y的函数 ，则$E(X|Y)$是Y的函数 。

举例：投硬币，正面朝上的概率为Y，投掷次数为n，正面朝上次数为X，则 $E(X|Y=y)=ny$，得出 $E(X|Y)=nY$ ，因此$E(X|Y)$也是随机变量。

##### 条件期望

离散型

$$E ( X | Y = y ) = \sum _ { x \in \mathcal X } x P ( X = x | Y = y ) = \sum _ {x \in \mathcal X} x \frac { P ( X = x , Y = y ) } { P ( Y = y ) }$$ 

连续型

$$E ( X | Y = y ) = \int _ { \mathcal X} x f_{\mathcal X} ( X | Y = y ) d x$$  

##### 条件期望的期望

离散型

$$E [ E [ X | Y ] ]$$$$=\sum _ { y } P_Y ( y )E [ X [ Y = y ] $$

连续型

$$E [ E [ X | Y ] ]$$$$=\int _ { - \infty } ^ { \infty } E [ X | Y = y ] f _ { Y } ( y ) d y$$

##### 迭代期望法则

又名重期望法则：$$E [ E [ X | Y ] ] = E [ X ]$$，即条件期望的期望等于无条件期望。（根据全期望定理可得出）



#### 方差

$Var(aX)=a^{2}Var(X)$ 



$$ { v a r } [ X ] = E [ ( X - E ( X ) ) ^ { 2 } ]$$

$$\qquad \quad = E [ X ^ { 2 } + E [ X ] ^ { 2 } - 2 X E [ X ] ]$$

$$\qquad \quad= E [ X ^ { 2 } ] + E [ X ] ^ { 2 } - 2 E [ X ] ^ { 2 }$$

$$\qquad \quad= E [ X ^ { 2 } ] - E [ X ] ^ { 2 }$$                          <span id="eq1">**【公式一】**</span> 



#### 协方差

$$ { c o v } [ X , Y ] = E [ ( X - E ( X ) ) ( Y - E ( Y ) ) ]$$

$$\qquad \qquad = E [ X Y - X E [ Y ] - E [ X ] Y + E [ X ] E [ Y ] ]$$

$$\qquad \qquad  = E [ X Y ] - E [ X ] E [ Y ] - E [ X ] E [ Y ] + E [ X ] E [ Y ] E [ Y ]$$

$$\qquad \qquad = E [ X Y ] - E [ X ] E [ Y ]$$                               <span id="eq2">**【公式二】**</span> 

同变量的协方差等于方差：$$cov [ X , X ] =  { v a r } [ X ]$$

两个分布独立的变量协方差为0。



### 随机变量点积

#### 期望

$$E [ X Y ] =E [ E [ X Y | Y ] ]$$$$= E [ Y  E [ X | Y ] ]$$                    迭代期望法则

$$E [ E [ X Y| Y ] ]$$$$=\sum _ { y } P_Y ( y )E [ Xy [ Y = y ] $$ 

$$\qquad \qquad \quad =\sum _ { y } P_Y ( y )yE [ X [ Y = y ] $$                    常数提前

$$\qquad \qquad \quad  = E [ Y  E [ X | Y ] ]$$ 

当X和Y分布独立$$E [ X | Y ] = E [ X ]$$，则：$$E [ X Y ] = E [ Y \cdot E [ X ] ]$$$$= E [ X ] \cdot E [ Y ]$$

#### 方差

$$ { v a r } [ X Y ] = E [ X ^ { 2 } Y ^ { 2 } ] - E [ X Y ] ^ { 2 }$$

根据[公式一](#eq1) 和[公式二](#eq2) ：

$$E [ X ^ { 2 } Y ^ { 2 } ] = cov [ X ^ { 2 } , Y ^ { 2 } ] + E [ X ^ { 2 } ] E [ Y ^ { 2 } ]$$ 

$$\qquad \qquad = \cos [ X ^ { 2 } , Y ^ { 2 } ] + ( E [ X ] ^ { 2 } + v a x [ X ] ) \cdot ( E [ Y ] ^ { 2 } + v a r [ Y ] )$$ 



$$E [ X Y ] ^ { 2 } = ( \cot X , Y ] + E [ X ] E [ Y ] ) ^ { 2 }$$  



则$$ { v a r } [ X Y ]$$ 等于

$$ { v a r } [ X Y ] = \cot [ X ^ { 2 } , Y ^ { 2 } ] + ( E [ X ] ^ { 2 } +  { v a r } [ X ] ) \cdot ( E [ Y ] ^ { 2 } +  { v a r } [ Y ] ) - ( c o v [ X , Y ] + E [ X ]E [ Y ] ) ^ { 2 }$$  

当X和Y分布独立$$\cos [ X ^ { 2 } , Y ^ { 2 } ] = \cos [ X , Y ] = 0$$，则：

$$ { v a r } [ X Y ] = ( E [ X ] ^ { 2 } +  { v a r } [ X ] ) \cdot ( E [ Y ] ^ { 2 } +  { v a r } [ Y ] ) - ( E [ X ] E [ Y ] ) ^ { 2 }$$ 

$$\qquad \qquad = E [ X ] ^ { 2 }  { v a r } [ Y ] + E [ Y ] ^ { 2 }  [ X ] +  { v a r } [ X ]  [ Y ]$$



因为在这里X和Y是以0为均值的，所以$$ { v a r } [ X Y ] =  { v a r } [ X ]  { v a r }  [ Y ]$$



### 随机变量的和

设Z是n个随机变量的和，$$Z = \sum _ { i = 1 } ^ { n } X _ { i }$$  

#### 期望

$$E [ Z ] = \sum _ { i = 1 } ^ { n } E [ X _ { i } ]$$

#### 方差

$$ { v a r } ( Z ) =  { c o v } [ \sum _ { i = 1 } ^ { n } X _ { i } \space  ,\space \sum _ { j = 1 } ^ { n } X _ { j }]$$ 

$$\qquad \quad  = \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } \cos [ X _ { i } , X _ { j } ]$$  



当$X_i$互相独立：

$$ { v a r } ( Z ) = \sum _ { i = 1 } ^ { n } cov [ X _ { i } , X _ { i } ]$$

$$\qquad \quad = \sum _ { i = 1 } ^ { n }  { v a r } [ X _ { i } ]$$



### 随机向量点击

设q和k是两个$d_k$维的向量，并且每一维是独立的，且

$$E [ q _ { i } ] = E [ k _ { i } ] = 0$$ 

$$ { v a r } [ q _ { i } ] =  { v a r } [ k _ { i } ] = 1$$ 

$$i \in [ 0 , d _ { k } ]$$

那么：

$$E [ q \cdot k ] = E [ \sum _ { i = 1 } ^ { d _ { k } } q _ { i } k _ { i } ]$$

$$\qquad \quad = \sum _ { i = 1 } ^ { d _ { k } } E [ q _ { i } k _ { i } )$$ 

$$\qquad \quad= \sum _ { i = 1 } ^ { d _ { k } } E [ q _ { i } ] E [ k _ { i } ]$$ 

$$\qquad \quad=0$$ 



$${ v a r } [ q \cdot k ] =  { v a r } [ \sum _ { i = 1 } ^ { d _ { k }  }q_ik_i ]$$ 

$$\qquad \quad= \sum _ { i = 1 } ^ { d _ { k } }  { v a r } [ q _ { i } k _ { i } ]$$ 

$$\qquad \quad= \sum _ { i = 1 } ^ { d _ { k } }  { v a r } [ q _ { i } ]  [ k _ { i } ]$$ 

$$\qquad \quad= \sum _ { i = 1 } ^ { d _ { k } }  { v a r } [ q _ { i } ]  [ k _ { i } ]$$ 

$$\qquad \quad= \sum _{i=1}^{d_{k}}1$$ 

$$\qquad \quad= d _ { k }$$  



则：

$$ { v a r } [\frac{q \cdot k }{\sqrt d_k} ] ={\frac{1 }{ d_k}}  { v a r } [ \sum _ { i = 1 } ^ { d _ { k }  }q_ik_i ]$$  $=1$







## 引用

1. [Statistical-Properties-of-Dot-Product/proof.pdf at master · BAI-Yeqi/Statistical-Properties-of-Dot-Product (github.com)](https://github.com/BAI-Yeqi/Statistical-Properties-of-Dot-Product/blob/master/proof.pdf)





<font size=24>关注本公众号，下期更精彩</font>

<img src="https://notebook-media.oss-cn-beijing.aliyuncs.com/img/image-20220930221129484.png" alt="image-20220930221129484" style="zoom: 80%;" />

