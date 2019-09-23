# Loss-Balanced Task Weighting

This repository contains the source code for the paper
> Shengchao Liu, Yingyu Liang, Anthony Gitter. [Loss-Balanced Task Weighting to Reduce Negative Transfer in Multi-Task Learning
](https://doi.org/10.1609/aaai.v33i01.33019977). AAAI-SA 2019.

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

[1] Wang, Yanli, Stephen H. Bryant, Tiejun Cheng, Jiyao Wang, Asta Gindulyte, Benjamin A. Shoemaker, Paul A. Thiessen, Siqian He, and Jian Zhang. "Pubchem bioassay: 2017 update." Nucleic acids research 45, no. D1: D955-D963, 2016.

[2] Chen, Zhao, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. "Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks." arXiv preprint arXiv:1711.02257, 2017.

[3] Liu, Shengchao. "Exploration on Deep Drug Discovery: Representation and Learning." University of Wisconsin-Madison, Master’s Thesis TR1854, 2018.

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
