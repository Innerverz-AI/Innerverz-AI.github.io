---
layout: post
title:  "[Paper Review] FSLSD: High-resolution Face Swapping via Latent Semantics Disentanglement"
date:   2022-05-02
author: 류원종
tags: [Deep Learning, Face Swapping, StyleGAN, Deep Fake]
---

![title](/assets/posts/face-swapping/FSLSD/title.PNG){: width="100%", height="100%"}<br>

# Motivation

MegaFS 의 결과를 보니 해상도가 떨어지고 skin tone 유지가 잘 안되네? 이건 entanglement 의 문제 같은데?

structure attributes (shape, pose, expression) 와 appearance attributes (illumination, skin tone) 를 latent space 상에서 분리하자!

# Overview

![scheme](/assets/posts/face-swapping/FSLSD/scheme.PNG){: width="100%", height="100%"}<br>

이 논문 역시 pretrained StyleGAN generator 를 이용하지만, StyleGAN 에서 나온 결과물은 side-ouptut swapped face 라고 하고, final swapped face 를 만드는 네트워크는 따로 둔다. 그 중간에서 나오는 feature 맵을 활용한다. target 의 appearance attibutes 는 그대로 사용하면서 source 와 target 의 structure attributes 를 섞는다. target 의 배경을 그대로 재현해기 위해, target 에서 multi-resolution feature 를 뽑아 StyleGAN generator 의 middle features 와 섞어준다. 그리고 이를 Decoder 에 넣어 final swapped face 를 생성한다. 

# Methods

## 3.1 Class-Specific Attributes Transfer
 
얼굴은 각기 다른 attributes class 를 포함한다. 예를들어, pose & expression 은 structure 에, lighting&colors 는 appearance 에 해당한다. 
face swapping 과정에서, structure attributes 는 source 와 target 양쪽을 섞어야 하지만, appearance attributes 는 target 만을 이용하면 된다. 

> structure attributes 를 섞는 이유로, 저자들은 source 의 identity 를 유지하기 위해서라고 하는데, 눈코입의 형태가 pose, expression 과 연관 있다는 논리인듯하다. 

StyleGAN W+ space 의 18 개 벡터 중 7개를 structure part $g \in R^{7\times512}$로 선택하고, 나머지는 appearance part $h \in R^{11\times512}$라고 한다. $w_s = (g_s, h_s)$ 와 $w_t = (g_t, h_t)$ 는 pSp encoder 를 이용해 얻는다. 결과적으로, swapped face 의 structure part 는 source 의 identity 와 target 의 pose&expression 을 포함해야 한다. source 와 target 의 structure part 를 섞을 때는 pose&expression 차이를 고려해야 하기 때문에, source structure attribute latent vector 를 편집할 때 이를 미리 고려해준다.

$$
\hat{g_s} = g_s + \vec{n}
$$

$\vec{n}$ 를 구하기 위해, facial landmark 를 이용한다. 

$$
\vec{n} = E_{le}(l_s, l_t)
$$

target 에서 swapped face 로 appearance attributes 를 옮기기 위해, $h_t$ 와 $\hat{g_s}$ 를 합친다.  

$$
\hat{w_s} = Concatenate(\hat{g_s}, h_t)
$$

$\hat{w_s}$ 는 StyleGAN generator 에 입력되어 side-output swapped face $y_s$ 를 만들어낸다.  


## 3.2  Background Transfer

Face Swapping applications 들을 위해 targe 의 배경을 잘 옮겨와야 한다. 
보통 Poisson Blending 을 이용해 inner-face 를 옮겨오지만, 이는 얼굴 외곽에 부자연스러운 경계선을 남긴다. 
이를 해결하기 위해, $y_s$ 는 side-output 으로 남겨두고, StyleGAN generator 에서 얻어낸 $F_s = \{f_s^0, f_s^1, ... , f_s^N\}$ 을 활용한다. 
비슷하게, target encoder $E_t$ 를 이용해 $F_t = \{f_t^0, f_t^1, ... , f_t^N\}$ 를 만들어낸다. 
inner-face mask $m_t$ 를 이용해 $f_t^i$ 와 $f_s^i$ 의 inner-face 부분을 교체한 뒤에, 이를 decoder 에 넣어 final face image $y_f$ 를 얻어낸다.

$$
y_f = Dec(F_t, F_s, m_t)
$$

이로써 자연스러운 배경 합성이 가능해지고, 동시에 $\hat{w_s}$ 가 facial region 과 attriutes transfer 에 집중하는게 가능해진다. 

## Objective

BeautyGAN 을 따라, Style-transfer loss $L_{st}$를 적용한다. histogram matching 을 이용해 guidance 를 만들어준다. 

$$\begin{aligned}


L_{lmk} &= ||E_{lmk}(y_s) - E_{lmk}(x_t)||_2 + ||E_{lmk}(y_f) - E_{lmk}(x_t)||_2 \\


L_{rec} &= ||y_f - x_t||_2 + ||y_s - x_t||_2 + 0.8 \times||F(y_f) - F(x_t)||_2 + 0.8 \times||F(y_s) - F(x_t)||_2 \\


L_{st} &= ||y_f - HM(y_f, x_t)||_2 \\


L_{total} &= L_{adv} + 2 \times L_{id} + 0.1 \times L_{lmk} + 2 \times L_{rec} + 0.2 \times L_{st} \\


\end{aligned}$$

## 3.3 Video Face Swapping

설명이 부족해 생략한다.

## Implementation Details 

FFHQ 데이터셋으로 학습한 1024x1024 resolution StyleGAN2 Generator 와 pSp Encoder 를 사용했다.
Landmark encoder 는 pSp 를 약간 변형해 사용했다. 

Optimizer: Adam $(\beta1=0.9, \beta2= 0.999, \epsilon=1e-8)$

Learning rate: 1e-4

Batch size: 8

Iterations: 500K

GPUs: 4-way Tesla V100 

Training time: two days for the whole model

## Dataset

Evaluated on CelebA-HQ and FaceForensics++.

# Results

## Comparison with existing methods

![result1](/assets/posts/face-swapping/FSLSD/result1.PNG){: width="100%", height="100%"}<br>

MegaFS 로 생성된 이미지는 흐리고 디테일이 떨어지며, 얼굴 외곽에 부자연스러운 경계선이 생긴다. 
또한 source 와 target 의 semantic gap 이 큰 경우 target attributes 를 잘 옮기지 못한다.
반면 이 연구의 결과는 얼굴의 디테일도 잘 유지하면서, target attributes 를 효과적으로 옮겨낸다. 

## ablation study

![result2](/assets/posts/face-swapping/FSLSD/result2.PNG){: width="100%", height="100%"}<br>

MegaFS 를 baseline 으로 삼아, 세 가지 방법의 차이를 보였다. Var.1 은 source 의 appearance attribute 를 유지했을 경우다. 결과에서 보이듯 illumination 이 섞여있다. Var.2 는 target 의 background 를 그대로 target face 에 섞어 final output 으로 사용한 경우다. 얼굴 경계가 부자연스럽다. 

## Face Swapping on High-resolution Videos

![result3](/assets/posts/face-swapping/FSLSD/result3.PNG){: width="100%", height="100%"}<br>

이미지마다 face swapping 을 적용하면 incoherent structure and appearance 가 보인다. 
trajectory constraint 는 그런 현상을 억제해준다. 

# Conclusion

StyleGAN 을 이용한 high-resolution face swapping 방법을 제시했다.
Attributes 를 structure 와 appearance 로 분류했고, disentangled latent space 에서 이들을 각각 옮겼다. 
structure attribute transfer 를 위해 landmark encoder 를 이용해 latent direction 을 예측했고,
StyleGAN 의 중간 결과들과 target image encoder 에서 얻은 multi-resolution features 를 합쳐 자연스러운 배경을 만들어냈다. 
두 가지 spatio-temporal constraints 를 이용해 video face swapping 으로 확장했다. 다만 Attributes transfer 가 latent space 에서 이뤄지는 만큼, GAN inversion method 의 성능에 영향을 많이 받는다. 또한 attributes 를 선택적으로 옮기지 못한다는 한계점이 있다. 

# My opinion

이 논문과 MegaFS 은 18 개의 latent code 중에서 어떤 부분을 섞는지가 다르다. MegaFS 는 18 개를 (4개, 14개) 로 나눈 뒤 fine detail 을 포함하는 14 개의 latent code 를 섞었다. 이 부분을 섞은 이유는 coarse information 부분에는 얼굴 구조, 각도의 정보가 포함되어 있기 때문에 source 의 것을 그대로 둔 채로 다른 부분만 섞으려 했기 때문이다. 이 논문에서는 18 개를 (7개, 11개) 로 나눈 뒤 coarse information 을 포함하는 7개 latent code 를 섞었다. MegaFS 와 비슷한 논리지만 반대의 생각으로, fine detail 부분은 target 의 것을 그대로 가져다 썼다. 그리고 두 이미지의 coarse information 을 섞었는데, 이는 눈코입 등 얼굴 요소들이 얼굴 각도와 연관이 있기 때문에 low level feature 까지 고려해야 한다고 생각한 것같다. 그래서 latent space 에서 이 정보를 변형하기 위해 facial landmark 를 인코딩해 버무려줬다. 아이디어는 그럴듯 하지만, facial landmark 를 적용한게 얼마나 효과있는지는 잘 모르겠다. 문제는, Face Swapping 에서 identity 가 제대로 옮겨진 것 같지 않다. pSp 의 성능 문제일수도 있고, latent vector 가 제대로 만들어지지 않았을 수도 있다. 그리고 video face swapping 쪽이 완전히 작성되지 않은 것 같다.


그리고 이 논문에서는 MegaFS 의 결과가 blurry 하다는 점을 지적하면서 attributes 사이에 entanglement 를 제대로 해결하지 않아서 그렇다고 하지만, 코드를 보면 그렇지 않다고 생각된다. MegaFS 의 결과가 blurry 한 이유는 1) adversarial loss 를 쓰지 않았고, 2) StyleGAN 의 결과에 4x pooling 을 사용해서 그런듯 하다.