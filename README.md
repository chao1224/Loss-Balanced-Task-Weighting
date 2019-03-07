# Loss-Balanced Task Weighting

This repository contains the source code for the paper
> Shengchao Liu, Yingyu Liang, Anthony Gitter. [Loss-Balanced Task Weighting to Reduce Negative Transfer in Multi-Task Learning
](https://www.aaai.org/Papers/AAAI/2019/SA-LiuS.371.pdf). AAAI-SA 2019.

The figshare appendix is public [here](https://doi.org/10.6084/m9.figshare.7732964).

This repository implements several multi-task learning algorithms, including the Loss-Balanced Task Weighting (LBTW) approach.
LBTW dynamically sets tasks weights while training a multi-task neural network.

### Setup

+ [Anaconda2-4.3.1](https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh)
+ PyTorch=0.3
+ scikit-learn=0.19

### Datasets

Randomly split PubChem BioAssay (PCBA) [1] into 5 folds.
PCBA has 128 binary classification tasks.

### Experiments

+ 128 Single-Task Learning (STL).
+ 1 Multi-Task Learning (MTL).
+ 128 Fine-Tuning.
+ 2 GradNorm [2].
+ 1 RMTL [3].
+ 2 Loss-Balanced Task Weighting (LBTW).

### Result

![Precision-Recall AUC results.](/image/pr.png)

### Reference

[1] Wang, Y., Bryant, S. H., Cheng, T., Wang, J., Gindulyte, A., Shoemaker, B. A., ... & Zhang, J. (2016). Pubchem bioassay: 2017 update. Nucleic acids research, 45(D1), D955-D963.

[2] Chen, Z.; Badrinarayanan, V.; Lee, C.-Y.; and Rabinovich, A. 2018. GradNorm: Gradient normalization for adaptiveloss balancing in deep multitask networks. InInternationalConference on Machine Learning, 793–802.

[3] Liu, S. 2018. Exploration on deep drug discovery: Repre-sentation and learning. Master’s Thesis TR1854.

### License

This project is released under the [MIT license](https://github.com/chao1224/Loss-Balanced-Task-Weighting/blob/master/LICENSE).

### Citation

```
@article{liu2019loss,
    title={Loss-Balanced Task Weighting to Reduce Negative Transfer in Multi-Task Learning},
    author={Liu, Shengchao and Liang, Yingyu and Gitter, Anthony},
    booktitle={Association for the Advancement of Artificial Intelligence (Student Abstract)},
    year={2019}
}
```
