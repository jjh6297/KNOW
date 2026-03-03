

# Learning from Oblivion: Predicting Knowledge-Overflowed Weights via Retrodiction of Forgetting

Official code release for **“Learning from Oblivion: Predicting Knowledge-Overflowed Weights via Retrodiction of Forgetting”**, **CVPR 2026** (to appear).


## Abstract

Pre-trained weights have become a cornerstone of modern deep learning, enabling efficient knowledge transfer and improving downstream task performance, especially in data-scarce scenarios. However, a fundamental question remains: how can we obtain better pre-trained weights that encapsulate more knowledge beyond the given dataset? In this work, we introduce **KNowledge-Overflowed Weights (KNOW)** prediction, a novel strategy that leverages structured forgetting and its inversion to synthesize knowledge-enriched weights. Our key insight is that sequential fine-tuning on progressively downsized datasets induces a structured forgetting process, which can be modeled and reversed to recover knowledge as if trained on a larger dataset. We construct a dataset of weight transitions governed by this controlled forgetting and employ meta-learning to model weight prediction effectively. Specifically, our **KNowledge-Overflowed Weights Nowcaster (KNOWN)** acts as a hyper-model that learns the general evolution of weights and predicts enhanced weights with improved generalization. Extensive experiments across diverse datasets and architectures demonstrate that KNOW prediction consistently outperforms Na\"ive fine-tuning and simple weight prediction, leading to superior downstream performance. Our work provides a new perspective on reinterpreting forgetting dynamics to push the limits of knowledge transfer.

![alt text](https://github.com/jjh6297/KNOW/blob/main/images/Schematic1.png?raw=true)




## Pre-trained Weights
Pre-trained weights of WNN are included.
'KNOWN_XXX.h5 ' in this repo are the pre-trained weights for each mathematical operation type (Conv, FC, Bias).


## Experiments

0. Metadata collection

```
python 0_Data_Collection_Sample.py
```


1. Pretraining with progressive forgetting

```
python 1_CIFAR100_Pretraininng.py
```

2. KNOW prediction

```
python 2_KNOW_Predict.py
```


3. Fine-tuning on the downstream task

```
python 3_Finetuning_KNOW_Predicted.py
```


4. Baseline: naive transfer learning

```
python 4_Finetuning_Baseline.py
```


## Citation

```
@article{jang2025learning,
  title={Learning from Oblivion: Predicting Knowledge Overflowed Weights via Retrodiction of Forgetting},
  author={Jang, Jinhyeok and Kim, Jaehong and Kim, Jung Uk},
  journal={arXiv preprint arXiv:2508.05059},
  year={2025}
}
```
