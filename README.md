# \[NeurIPS2023\] Exploring Diverse In-Context Configurations for Image Captioning
[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/y2l/meta-transfer-learning-tensorflow/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9-blue.svg?style=flat-square&logo=python&color=3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0.1-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/)

This repository contains the PyTorch implementation for the NeurIPS 2023 Paper ["Exploring Diverse In-Context Configurations for Image Captioning"](https://nips.cc/virtual/2023/poster/71057) by [Xu Yang](https://yangxuntu.github.io/), Yongliang Wu, Mingzhuo Yang, Haokun Chen and Xin Geng.

If you have any questions on this repository or the related paper, feel free to create an issue. 

#### Summary

* [Introduction](#introduction)
* [Getting Started](#getting-started)
* [Datasets](#datasets)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)


## Introduction
After discovering that Language Models (LMs) can be good in-context few-shot learners, numerous strategies have been proposed to optimize in-context sequence configurations. Recently, researchers in Vision-Language (VL) domains also develop their few-shot learners, while they only use the simplest way, e.g., randomly sampling, to configure in-context image-text pairs. In order to explore the effects of varying configurations on VL in-context learning, we devised four strategies for image selection and four for caption assignment to configure in-context image-text pairs for image captioning. Here Image Captioning is used as the case study since it can be seen as the  visually-conditioned LM. Our comprehensive experiments yield two counter-intuitive but valuable insights, highlighting the distinct characteristics of VL in-context learning due to multi-modal synergy, as compared to the NLP case.


![My SVG Image](doc/intro.svg)


> Figure: The distinction between LM and VLMs as few-shot learners. LM generally excel with examples akin to the test case (blue blocks in (a)). In contrast, for VLMs, the performance is not strictly correlated with image similarity but heavily relies on the caption quality. For instance, when low-quality captions are used, similar images (d) lead to worse performance than dissimilar ones (f) since VLMs may build a short-cut by reusing in-context captions without seeing the given images.

## Getting Started

Create a conda environment for running the scripts, run
```bash
conda create -n of python=3.9
pip install -r requirements.txt
pip install -e .
```

Download the OpenFlamingo v1 9B model from [link](https://huggingface.co/openflamingo/OpenFlamingo-9B-deprecated) and then download the LLaMA model from [link](https://huggingface.co/decapoda-research/llama-7b-hf).

You can run the following command to validate the model. See run_eval.sh for more details.

```bash
python open_flamingo/eval/evaluate.py \
    --lm_path $LM_PATH \
    --lm_tokenizer_path $LM_TOKENIZER_PATH \
    --checkpoint_path $CKPT_PATH \
    --device $DEVICE \
    --coco_image_dir_path $COCO_IMG_PATH \
    --coco_annotations_json_path $COCO_ANNO_PATH \
    --mgc_path  "MGC/wc_vis_135.json"\
    --mgca_path  "MGCA-idx/best_gt_WC(135).json"\
    --clip_ids_path "train_set_clip.json"
    --results_file $RESULTS_FILE \
    --num_samples 5000 --shots 4 8 16 32 --num_trials 1 --seed 5 --batch_size 8\
    --cross_attn_every_n_layers 4\
    --eval_coco
```


## Datasets


### MSCOCO
COCO is a large-scale object detection, segmentation, and captioning dataset. For image caption task, it has 5 captions per image. You can download the dataset from [link](https://cocodataset.org/#download).


## Citation

Please cite our paper if it is helpful to your work:

```bibtex
@article{yang2023exploring,
  title={Exploring Diverse In-Context Configurations for Image Captioning},
  author={Yang, Xu and Wu, Yongliang and Yang, Mingzhuo and Chen, Haokun and Xin, Geng},
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
