$$\text{FFN}(x) = (x \cdot W_{\text{gate}} \cdot \sigma_{\text{SiLU}}(x \cdot W_{\text{gate}})) \odot (x \cdot W_{\text{up}}) \cdot W_{\text{down}}$$
其中：
$x \in \mathbb{R}^{d}$ : 输入向量 (d=2560)
$W_{\text{gate}} \in \mathbb{R}^{d \times d_{\text{ff}}}$ : 门控投影权重 (2560×9728)
$W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ff}}}$ : 上投影权重 (2560×9728)
$W_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d}$ : 下投影权重 (9728×2560)
$\sigma_{\text{SiLU}}(x) = x \cdot \text{sigmoid}(x)$ : SiLU激活函数
$\odot$ : 逐元素相乘（Hadamard积）