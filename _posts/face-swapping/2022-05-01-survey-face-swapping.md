---
layout: post
title:  "[Survey] Deep Face Swapping"
date:   2022-05-01
author: 류원종
tags: [Deep Learning, Face Swapping, StyleGAN]
---

Face swapping 은 entertainment, human-computer interaction 등에서 좋은 목적으로 사용되기도 하지만, deep fake 등 정치, 경제적으로 약용되기도 한다. 

Face swapping 을 위해 이미지 두 장(source & target)을 입력 받는데, source 는 identity 를, target 은 pose, expression, illumination, background 등을 제공하는 이미지를 의미한다. 

현존하는 Face Swapping 방법은 크게 두 부류로 나눌 수 있다. 첫 번째는 대상이 정해져 있는 subject-specific, 두 번째는 임의의 사람을 대상으로 하는 subject-agnostic 이다. 

1. subject-specific [11, 27, 34]
- source 와 target 으로 정해진 쌍이 있다. 
2. subject-agnostic [35, 5, 28, 38, 36]
- 임의의 source 와 target 에 적용한다.

StyleGAN 을 필두로 고해상도 이미지 생성이 대세인 가운데, face swapping 역시 1024x1024 크기를 타겟으로 하는 논문들이 등장했다. 
High-resolution swapped face 를 생성하기 어려웠던 이유는 3가지 인데, 
1. 인코더 단에서 정보 손실이 많기 때문에
2. GAN 학습이 불안정하기 때문에
3. GPU memory 의 한계 때문에  


[11]  https://github.com/ondyari/FaceForensics/tree/master/dataset/DeepFakes
[27] 
[34] [Disney (2020)] (https://studios.disneyresearch.com/2020/06/29/high-resolution-neural-face-swapping-for-visual-effects/)

[35] Towards open-set identity preserving face synthesis. 
[5] [FSnet](https://tatsy.github.io/projects/natsume2018fsnet/)
[28] Advancing high fidelity identity swapping for forgery detection.
[38] Face identity disentanglement via latent spacemapping
[36] Fast Face-swap Using Convolutional Neural Networks




