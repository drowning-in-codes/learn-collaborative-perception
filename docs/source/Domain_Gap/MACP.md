### MACP: Efficient Model Adaptation for Cooperative Perception

#### abstract

车对车（V2V）通信通过信息共享实现了 "看穿遮挡物"，极大地增强了联网和自动驾驶车辆（CAV）的感知能力，从而显著提高了性能。然而，当现有的单个代理模型显示出卓越的泛化能力时，从头开始开发和训练复杂的多代理感知模型可能既昂贵又没有必要。在本文中，我们提出了一个名为 MACP 的新框架，它能使预先训练好的单个代理模型具备合作能力。为了实现这一目标，我们**确定了从单一代理转向合作设置所面临的关键挑战，并通过冻结大部分参数和添加一些轻量级模块来调整模型**。我们在实验中证明，所提出的框架可以有效地利用合作观察，并在模拟和真实世界合作感知基准中优于其他最先进的方法，同时所需的可调参数大大减少，通信成本也降低了。

#### 方法

我们的目标是求解一个最佳模型 f ∗，该模型能够检测和划定周围物体的边界框，并分配适当的标签。

为了简化符号，我们用一个 d′维向量 yj∈Rd′ 来表示每个边界框及其类别标签。

在不失一般性的前提下，物体检测模型 f 是一个从点云空间到边界框及其标签的联合空间 f : X → Y 的映射，经过训练的模型理想地描述了以观测点云集 x 为条件观测边界框集 y 的概率，其值为
$$
p(\mathbf{y}|\mathbf{x};f)=\frac{p(\mathbf{x},\mathbf{y})}{p(\mathbf{x})} 
$$
如果我们用 pS (x) 表示在单个代理感知中观察到点云集的边际概率，用 pC(x) 表示在合作感知中观察到精确点云集的概率，由于 V2V 通信共享了额外的点云，这两个概率可能不同，即 pS (x) ̸= pC(x) 。

预训练模型给出的点云和边界框的联合分布偏离合作环境下的地面实况联合分布
$$
\hat{p}_{\mathcal{C}}(\mathbf{x},\mathbf{y};f)=\frac{p_{\mathcal{S}}(\mathbf{x},\mathbf{y})}{p_{\mathcal{S}}(\mathbf{x})}\cdot p_{\mathcal{C}}(\mathbf{x})\neq p_{\mathcal{C}}(\mathbf{x},\mathbf{y}) \\
g^*=\underset{g\in\mathcal{G}}{\text{argmin}\mathcal{L}}\left(p_{\mathcal{C}}(\mathbf{x},\mathbf{y}),\hat{p}_{\mathcal{C}}(\mathbf{x},\mathbf{y};f\cdot g)\right) \\
p(\mathbf{y}|\mathbf{x};f\cdot g)=g\left\lfloor\frac{p_{\mathcal{S}}(\mathbf{x},\mathbf{y})}{p_{\mathcal{S}}(\mathbf{x})}\right\rfloor 
$$
![image-20240508151535055](https://s2.loli.net/2024/05/08/GZMfEleua21jYVv.png)

#### Convolution Adapter

ConAda 模块是特征编码器的关键组件。特征编码器网络是卷积块的级联，其中卷积层的输出经过 ConAda 模块，并通过残差连接加回自身。我们只在训练过程中训练 ConAda 参数，并在卷积层和 ConAda 模块之后的其他层中冻结预训练参数。

同时，ConAda 还充当车辆之间的通信通道。在通信过程中，ConAda 模块中的下卷积层和激活层帮助压缩和加密编码特征，以便进行广播，而上卷积层则用于解压缩接收信号，以便进行特征融合。

#### SSF Operator for Fused Feature

我们在连续的神经网络层中执行 SSF 算子，以考虑域偏移.

假设卷积层的输出特征图由 $X^{output}_{i,j}$ ∈ $R^{H′×W ′×C′}$ 给出，我们使用缩放因子 $\gamma$∈ $R^{C′}$和移动因子 β ∈ $R^{C′}$更新特征图
$$
X_{i,j}^{\mathrm{output}}=\gamma\odot X_{i,j}^{\mathrm{output}}+\beta 
$$
最后,基于 ConAda 的通信信道可以灵活压缩信号传输,从而缓解通信瓶颈.

![image-20240508165439383](https://s2.loli.net/2024/05/08/AjeXstVr2Ih6E5g.png)

这篇文章主要基于微调的方法和思想,使用的是Adapter,此外还有LoRA等.关于大模型压缩技术还有剪枝、蒸馏以及量化等等,感觉都可以试试.