# VS-DARTS (accepted at BMVC-21)
Abstract: Differentiable architecture search (DARTS) has become the popular method of neural architecture search (NAS) due to its adaptability and low computational cost. However, following the publication of DARTS, it has been found that DARTS often yields a suboptimal neural architecture because architecture parameters do not accurately represent operation strengths. Through extensive theoretical analysis and empirical observations, we reveal that this issue occurs as a result of the existence of unnormalized operations. Based on our finding, we propose a novel variance-stationary differentiable architecture search (VS-DARTS), which consists of node normalization, local adaptive learning rate, and sqrt(β)-continuous relaxation. Comprehensively, VS-DARTS makes the architecture parameters a more reliable metric for deriving a desirable architecture without increasing the search cost. In addition to the theoretical motivation behind all components of VSDARTS, we provide strong experimental results to demonstrate that they synergize to significantly improve the search performance. The architecture searched by VS-DARTS achieves the test error of 2.50% on CIFAR-10 and 24.7% on ImageNet.

This code is mainly based on DARTS (https://github.com/quark0/darts).

# Major Dependencies & Tested Environments
- python 3.8
- PyTorch 1.7.1 
- NVIDIA-apex (please install from here: https://github.com/NVIDIA/apex)
- Ubuntu 16.04, 18.04, 20.04

# For searching architecture with VS-DARTS in CIFAR-10
python train_search.py

# For train searched architecture from scratch(CIFAR-10).
python train.py

# For evaluating pretrained model(founded by VS-DARTS) in CIFAR-10.
python test.py 

# If our code is useful, please cite this paper.
pdf: https://www.bmvc2021-virtualconference.com/assets/papers/0549.pdf

BMVC Virtual conference link: https://www.bmvc2021-virtualconference.com/conference/papers/paper_0549.html

Paper: Choe, Hyeokjun, et al., "Variance-stationary Differentiable NAS.", the 32nd British Machine Vision Conference, 2021.
