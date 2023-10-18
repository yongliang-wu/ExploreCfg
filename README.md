# Exploring Diverse In-Context Configurations for Image Captioning
[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/y2l/meta-transfer-learning-tensorflow/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9-blue.svg?style=flat-square&logo=python&color=3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0.1-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/)

This repository contains the PyTorch implementation for the NeurIPS 2023 Paper ["Exploring Diverse In-Context Configurations for Image Captioning"](https://nips.cc/virtual/2023/poster/71057) by [Xu Yang](https://yangxuntu.github.io/), Yongliang Wu, Mingzhuo Yang, Haokun Chen and Xin Geng.

If you have any questions on this repository or the related paper, feel free to create an issue. 

#### Summary

* [Introduction](#introduction)
* [Getting Started](#getting-started)
* [Datasets](#datasets)
* [Performance](#performance)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)


## Introduction

Meta-learning has been proposed as a framework to address the challenging few-shot learning setting. The key idea is to leverage a large number of similar few-shot tasks in order to learn how to adapt a base-learner to a new task for which only a few labeled samples are available. As deep neural networks (DNNs) tend to overfit using a few samples only, meta-learning typically uses shallow neural networks (SNNs), thus limiting its effectiveness. In this paper we propose a novel few-shot learning method called ***meta-transfer learning (MTL)*** which learns to adapt a ***deep NN*** for ***few shot learning tasks***. Specifically, meta refers to training multiple tasks, and transfer is achieved by learning scaling and shifting functions of DNN weights for each task. We conduct experiments using (5-class, 1-shot) and (5-class, 5-shot) recognition tasks on two challenging few-shot learning benchmarks: 𝑚𝑖𝑛𝑖ImageNet and Fewshot-CIFAR100. 

<p align="center">
    <img src="https://mtl.yyliu.net/images/ss.png" width="400"/>
</p>

> Figure: Meta-Transfer Learning. (a) Parameter-level fine-tuning (FT) is a conventional meta-training operation, e.g. in MAML. Its update works for all neuron parameters, 𝑊 and 𝑏. (b) Our neuron-level scaling and shifting (SS) operations in meta-transfer learning. They reduce the number of learning parameters and avoid overfitting problems. In addition, they keep large-scale trained parameters (in yellow) frozen, preventing “catastrophic forgetting”.

## Getting Started

Create a conda environment for running the scripts, run
```bash
conda create -n of python=3.9
pip install -r requirements.txt
pip install -e .
```

## Datasets


### MSCOCO
COCO is a large-scale object detection, segmentation, and captioning dataset. For image caption task, it has 5 captions per image. You can download the dataset from [link](https://cocodataset.org/#download).


## Citation

Please cite our paper if it is helpful to your work:

```bibtex
@article{yang2023exploring,
  title={Exploring Diverse In-Context Configurations for Image Captioning},
  author={Yang, Xu and Wu, Yongliang and Yang, Mingzhuo and Chen, Haokun},
  journal={arXiv preprint arXiv:2305.14800},
  year={2023}
}
```

## TODO
1. Add the implement of Model-Generated Captions
2. Add the implement of Model-Generated Captions as Anchors
3. Add the implement of Similarity-basedImage-ImageRetrieval, Similarity-basedImage-CaptionRetrieval and Diversity-basedImage-ImageRetrieval

## Acknowledgements

Our implementations use the source code from the following repository:

* [OpenFlamingo](https://github.com/mlfoundations/open_flamingo/tree/main)
