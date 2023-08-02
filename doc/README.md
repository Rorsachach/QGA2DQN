# 基于强化学习和量子遗传算法的加边策略研究

# 问题建模
本研究的目标是在相依网络的一侧添加固定数量的边来提高相依网络的鲁棒性。首先我们将该过程描述为带限制的01整数规划模型。

假设相依网络$G=<V,\ E>$由两部分组成，分别是加边侧网络$G_1=<V_1,E_1>$和相依侧网络$G_2=<V_2,E_2>$，则有：$G=G_1+G_2+E_{inter}$。其中$E_{inter}$是加边侧网络与相依侧网络之间一一对应的相依边。

在加边侧网络中，假设所有可加边的位置为$E_{ij}=\left(V_1^i,V_1^j\right)\ \ \ s.t.\ i\neq j,E_{ij}\notin E_1$，记可加边位置的长度$L$。在所有可加边位置中选取$l$个位置进行加边，使用一个二进制变量来描述加边侧网络的$i$和$j$节点之间是否进行加边，则可以得到实际加边：
$$
e_{ij}=x_{ij}\times E_{ij}\ \ \ \ \ s.t.\ i\neq j,E_{ij}\notin E_1,\sum x_{ij}=l
$$

然后，我们确定优化目标，即网络鲁棒性的评价方案。我们选用一种基于网络巨分量变化的鲁棒性评价指标$R$作为优化目标，该指标反映了网络中节点被移除后网络中巨分量的平均值。
$$
R=\frac{1}{N}\cdot\sum_{q=1}^{N}\frac{s\left(q\right)}{N}
$$

其中，N是网络中的节点数量，s\left(q\right)是第q个节点随机故障之后网络巨分量的大小，R值越大网络鲁棒性越好。

则问题可以被建模为：
$$
maxargR\left(G^\prime\right)
s.t.{\ G}_1^\prime=G_1+e_{ij},G^\prime=G_1^\prime+G_2
$$

在0-1规划模型基础上，我们引入量子遗传算法进行求解，并将其过程描述为马尔可夫决策过程，使用强化学习手段来指导量子遗传中种群迭代过程。

首先，我们对量子遗传算法求解该模型进行介绍。量子遗传算法是在遗传算法的基础上引入量子态编码、量子旋转门等概念来加速收敛速度。

量子遗传有几个关键概念：量子编码、量子旋转门和适应度函数。

量子编码
量子遗传算法基于量子比特进行计算。1个量子比特的状态可以被描述为$|\psi=\alpha|0+\beta|1$。其中$\alpha,\ \beta$是量子比特为0|1的概率幅度，$\left|\alpha\right|^2,\left|\beta\right|^2$分别为量子比特为$|0$态和$|1$态的概率，并且满足$\left|\alpha\right|^2+\left|\beta\right|^2=1$。

一个量子编码则由多个量子比特来组成，假设量子编码长度为n，则其对应的量子编码为：
$$
q=\left[\begin{matrix}\alpha_1\\\beta_1\\\end{matrix},\ \ldots,\begin{matrix}\alpha_n\\\beta_n\\\end{matrix}\ \right]\ 
$$
在该问题下，我们对所有可加边位置进行编码。

量子旋转门
量子遗传算法使用量子旋转门的方式进行种群的更新，量子旋转门通过改变量子叠加态的概率幅度来改变量子比特的状态。对该操作进行定义：
$$
\left[\begin{matrix}\alpha_i^\prime\\\beta_i^\prime\\\end{matrix}\right]=U\left(\theta\right)\cdot\left[\begin{matrix}\alpha_i\\\beta_i\\\end{matrix}\right]=\left[\begin{matrix}\cos{\theta_i}&-\sin{\theta_i}\\\sin{\theta_i}&\cos{\theta_i}\\\end{matrix}\right]\cdot\left[\begin{matrix}\alpha_i\\\beta_i\\\end{matrix}\right]
$$
传统的量子遗传算法通过预先人工定义量子旋转门表来进行量子比特状态的更新，策略表定义的好坏影响了最终收敛效率与收敛结果。本方案使用强化学习来指定旋转策略，策略制定更加细粒度，能更快的指导量子遗传编码进行收敛。

适应度计算
量子遗传算法的适应度函数用于评价解空间内的每个解的优劣程度，适应度越高则可行解越优。本方案采用网络的鲁棒性评价指标作为适应度函数。

变异


我们在量子遗传算法的基础上对该问题进一步建模为马尔可夫决策过程，并使用强化学习手段来进行求解。马尔可夫决策过程由五元组进行描述$<S,\ A,\ P,\ r,\ \ \gamma>$，具体的设计细节如下：

- 状态$S$：我们将量子遗传算法的量子编码作为状态使用。
- 动作$A$：我们采用量子旋转门的旋转角度作为动作，并使用连续动作空间$\left[-0.05\cdot\pi,0.05\cdot\pi\right]$作为动作取值范围。
- 状态转移函数$P$：我们在状态转移函数中引入量子遗传算法的编译操作，即对于量子编码的某个量子比特，以变异概率p进行状态对调，并以$1-p$的概率直接进行量子旋转操作。
- 奖励函数$r$：每一步操作都以适应度变化作为奖励值，即$fitness(G’) – fitness(G)$
折扣因子\gamma：以0.96作为折扣因子使用。




输入：当前量子遗传编码 gene, 可选择加边数量l，所有可选择的位置数L
solution=L¬
For i=1:L
  If random\left(0,1\right)<gene\left[i,\ \ 0\right]^2
    solution\left[i\right]=0
  Else
    solution\left[i\right]=1
  End
End
If len(find(solution==1))>1¬
  按照gene[:,0]从高到低排序，并按照排序将solution¬中的l_{find}-l个对应位置置为0
Else
  按照gene[:,0]从低到高排序，并按照排序将solution¬中的l-l_{find}个对应位置置为1
End
输出：solution

输入：量子遗传编码 gene，当前网络 network
fitness=0
For i=1:100
  通过表1来获取gene对应的一个确定个体solution
  tmp=0
  For j=1:100
    attackNodeList\ =\ shuffle\left(1:N\right)
    R\ =\ 0
For q=1:N
  network\ -=\ attackNodeList\left[q\right]
  R\ +=\ s\left(q\right)
End
    tmp\ +=\ R\ /\ 100
End
  fitness\ +=\ tmp\ /\ 100
End 
fitness\ =\ fitness\ /\ 100
输出：fitness

输入：执行动作action
episode\ +=\ 1
new\ state\ =\ action\cdot state
通过表2计算new\ state的fitness
reward=fitness\left(newstate\right)–fitnessstate
terminated=bool\left(episode>200\right)
state\ =\ new\ state
输出：下一状态 state, 奖励值 reward，是否终止done

输入：相依网络 network
初始化环境 env
初始化Agent\ agnet
For episode\ =\ 1:100
初始状态 state=env.reset
\ \ \ \ States=list
\ \ \ \ Actions\ =\ list
\ \ \ \ Rewards\ =\ list
For done
  Agent 通过当前环境状态 state 做出动作决策 action
  Next\ state,reward,done\ =\ env.step\left(action\right)
  记录 state,\ action,\ next\ state,\ reward
End 
使用这一次随机采样对来对agent进行更新
\ \ \ \ G\ =\ 0
For i=len\left(states\right):1
  计算当前状态累计收益: G\ =\ gamma\ \ast\ G\ +\ reward
  计算当前状态state下决策出对应action的概率点对数logprob
  Loss\ =\ -logprob\ \ast\ G
  计算梯度
End
进行梯度下降
End

