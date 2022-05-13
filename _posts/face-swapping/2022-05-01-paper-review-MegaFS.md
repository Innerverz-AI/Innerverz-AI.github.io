---
layout: post
title:  "[Paper Review] MegaFS: One Shot Face Swapping on Megapixels"
date:   2022-05-01
author: 류원종
tags: [Deep Learning, Face Swapping, StyleGAN, Deep Fake]
---

![title](/assets/posts/face-swapping/MegaFS/title.PNG){: width="100%", height="100%"}<br>

CVPR 2021 에 게재된 논문이다. StyleGAN 을 이용해 high-resolution face swapping 을 구현했다.

# Motivation
Face Swapping 기술이 좋긴한데 해상도가 너무 낮네?

StyleGAN 을 이용해 고해상도 face swapping 이미지를 생성해보자!

# Methods

![scheme](/assets/posts/face-swapping/MegaFS/scheme.PNG){: width="100%", height="100%"}<br>

Face swapping 의 목표는 source image $x_s$ 로부터 identity 를, target image $x_t$ 로부터 attributes 를 추출한 뒤, 이들을 합친 $y_{s2t}$ 를 생성하는 것이다. 

이 과정을 세 단계로 구분해서 생각해볼 수 있다.
- Stage I. Face Encoding
  - $x_s$ 와 $x_t$ 를 각각 latent vector $S \in W^{++}$ 로 표현한다. 
    - 부자연스러운 image-level image editing 이 아닌, 자연스러운 이미지를 생성할 수 있는 latent-level manipulation 이 가능해진다.
  - Latent vector $S$는 $C \in R^{4 \times 4\times 512}$ , $L^{low} \in R^{4 \times 512}$ , $L^{high} \in R^{14 \times 512}$ 로 구성된다. 
  - $L^{low}$ 는 이미지의 sparse information 을, $L^{high}$는 이미지의 fine information 을 포함한다. 
- Stage II. Latent Code Manipulation
  - Swapped face $y_{s2t}$ 의 정보를 담은 latent vector, $S_{s2t}$ 를 만든다.
  - $S_t$ 의 $C_t$ 와 $L_t^{low}$ 를 그대로 가져와 $S_{s2t}$ 의 $C$ 와 $L^{low}$로 사용한다.
  - $L_s^{high}$ 와 $L_t^{high}$ 둘을 섞어 $L_{s2t}$ 를 만들고, 이를 $S_{s2t}$ 의 $L^{high}$로 사용한다. 
- Stage III. Face Generation
  - StyleGAN2 generator 에 $S_{s2t}$ 를 입력해 swapped face 를 생성한다.

이 논문에서는 Stage I 에서 Hierarchial Represntation Face Encoder 를, Stage II 에서 Face Transfer module 을 제안한다. Stage III 에서는 StyleGAN2 generator 를 그대로 사용한다.

## 3.1 Hierarchial Representation Face Encoder (HieRFE)

![title](/assets/posts/face-swapping/MegaFS/HieRFE.PNG){: width="100%", height="100%"}<br>

Face swapping 의 첫 단계는 이미지로부터 정보를 추출하는 것이다. 이를 위해 저자들은 feature pyramid strcture 를 이용한 Hierarchial Encoder 를 제안한다. 이미지를 HieRFE 에 입력하면 Residual Block 들을 통과하며 feature pyramid 를 생성한다. 각 단계의 feature map 들은 mapping network 를 통과해 18 개의 single latent vector 로 표현된다. 그 중 일부는 4x512 크기의 low-level topology information $L^{low}$ 로, 나머지는 14x512 크기의 high-level semantic information $L^{high}$ 로 분류된다.

> 
이 방법은 pixel2Style2pixel 에서 제안하는 방법과 비슷하지만, styleGAN 에서 상수로 고정되어 있는 initial feature map $C \in R^{4 \times 4\times 512}$ 까지 구해낸다는 점이 다르다.
>

## 3.2 Face Transfer Module (FTM) 

![title](/assets/posts/face-swapping/MegaFS/FTB.PNG){: width="100%", height="100%"}<br>

위 3.1 에서 두 이미지의 정보를 추출했으니 이제 둘을 잘 섞어줄 차례다. $L^{low}$ 는 얼굴의 방향, 얼굴의 구조 등 굵직한 정보를 담당하고 있는데, swapped face 의 얼굴형이나 얼굴 방향은 target image 를 따라야 하므로 $L_{target}^{low}$ 를 사용한다. $L^{high}$ 는 눈코입의 형태, 표정, 색감 등 다양한 정보를 포함하고 있는데, 눈코입 등의 형태는 source image 를, 표정과 조명 정보는 target image 를 따라야 하므로 $L^{high}$ 는 둘을 섞어야 한다. 

두 이미지로부터 추출한 $L_t^{high}$ 와 $L_s^{high}$ 를 잘 섞어 swapped face 를 표현할 수 있는 representation vetor $L_{s2t}^{high}$ 를 얻어내기 위해 Face Transfer Moduel (FTM) 을 제안했다. FTM 은 $L^{high}$의 각 레이어별로 하나씩 Face Transfer Block (FTB) 를 두어 총 14개의 FTB 를 포함한다. FTB 는 $l_t^{high}$ 와 $l_s^{high}$ 를 하나씩 입력받아 $l_s2t$를 만들어낸다. FTB 는 Transfer cell 3개로 이루어져 있는데, 위 그림처럼 두 벡터가 입력으로 들어간다. Transfer cell 의 두 입력값을 각각 a, b 라고 하면, 아래와 같은 연산을 통해 출력값을 계산한다.

$$
c = concat(a, b)
$$

$$
a' = TanH(MN_2(Sigmoid(MN_1(c) \times a)) + a) 
$$

$$
b' = TanH(MN_2(Sigmoid(MN_1(c) \times b)) + b) 
$$


마지막으로, trainable weight $w \in R^{1 \times 512}$ 를 이용해 $l_{s2t}$ 를 얻는다.

$$
l_{s2t} = \sigma(w)\hat{l}_t^{high} + (1-\sigma(w)) \hat{l}_s^{high}
$$

그리고 14개의 $l_{s2t}$ 를 모아 $L_{s2t}$ 를 완성한다. 

## 3.3 High-Fidelity Face Generation

StyleGAN 을 그대로 사용하긴 한다. 자신들이 제안한 모듈들이 styleGAN 의 latent space 를 확장했기 때문에 얼굴을 표현하는 능력이 좋아지고 학습 과정이 효율적이라고 하는데, 논문에서 주장하는 설명이 논리적이지 않아 생략한다. 

# Training

이 논문에서 제안한 두 모듈은 순차적으로 학습한다. 즉 two stage learning 이다. 

## Objective functions of HieRFE

이전 논문들처럼 $L_{rec}$, $L_{LPIPS}$, $L_{id}$ 를 사용한다. 추가로 pose 와 expression 에 가이드를 주기 위해 landmark loss 를 추가한다. 

$$
L_{ldm} = ||P(x) - P(\hat{x})||_2
$$


## Objective functions of FTM

HieRFE 를 학습할 때와 마찬가지로 $L_{rec}$, $L_{LPIPS}$, $L_{id}$, $L_{ldm}$ 을 사용하며, 학습과정을 안정화하기 위해 $L_{norm}$ 를 추가한다.

$$
L_{norm} = ||L_s^{high} - L_{s2t}||_2
$$

## Dataset

CelebA, CelebA-HQ, FFHQ, FaceForensics++ 등을 사용했다.
StyleGAN 으로 200k 장의 이미지를 생성해 보조적으로 사용했다. 

## Details 

Learning rate 은 0.01 로 설정했다. HieRFE 와 FTM 은 순차적으로 10 epoch 씩 학습했다. 학습에는 CelebA, CelebA-HQ, FFHQ, 그리고 추가로 제작한 보조 데이터를 이용했다. 학습 시간은 Tesla V100 GPU 를 3대 써서 5일이 걸렸다.

# Results

## MegaFS

![result2](/assets/posts/face-swapping/MegaFS/result2.PNG){: width="100%", height="100%"}<br>

다양한 포즈와 표정에도 변환이 잘 된다.

## Comparison with existing methods

![result1](/assets/posts/face-swapping/MegaFS/result1.PNG){: width="100%", height="100%"}<br>

Faceshifter 와 MegaFS 가 가장 좋은 성능을 보이는데, Faceshifter 의 경우 face encoder 에서 뽑혀진 정보는 identity 와 다른 attributes 가 완벽히 disentangled 되어 있지 않다. 

## ablation study: W++ space

![result3](/assets/posts/face-swapping/MegaFS/result3.PNG){: width="100%", height="100%"}<br>

Reconstruction 테스트 결과인데, W+ space 를 쓰는 경우 sunglass, eyeglasses, eye gaze 등을 재현하는데 실패했다. 

## ablation study: FTM

![ftm](/assets/posts/face-swapping/MegaFS/ftm.PNG){: width="100%", height="100%"}<br>

이 논문에서는 $L_t^{high}$ 와 $L_s^{high}$ 를 잘 섞어 $L_{s2t}^{high}$ 를 만드는 방법을 제안했는데, 이 방법에 대한 유효성을 검증한다. 이를 위해 세 가지 방법을 비교했는데, 첫 번쨰는 $L_t^{high}$ 와 $L_s^{high}$ 를 섞지 않고 $L_s^{high}$ 만을 사용하는 것이다. 이를 latent code replacement (LCR) 이라고 한다. 두 번째는 아래 그림처럼 self-attention 방식을 이용한 것으로, ID injection 이라고 부른다. 세 번째는 이 논문에서 제안한 FTM 을 사용하는 방식이다. 

위 그림에서 결과를 볼 수 있듯, LCR 방식은 target image 의 appearance 정보를 잃어버렸기 때문에 피부 색이 target 과 다르다. ID injection 은 target 의 expression 을 가져오는데 가장 효과적이었다. FTM 은 ID similarty 가 가장 높고, pose error 가 가장 낮았다. 

![idinjection](/assets/posts/face-swapping/MegaFS/idinjection.PNG){: width="100%", height="100%"}<br>


# Conclusion

- HieRFE: $W^{++}$ space 를 이용해 얼굴 복원 능력을 높였다. 

- FTM: id loss 외에 추가적인 loss function 없이 identity 를 잘 옮겨왔다. 

- Image generation: StyleGAN 을 이용해 고해상도 이미지를 생성할 수 있었고, 동시에 학습 시간과 메모리 사용량을 줄였다.

# My opinion

- 저자들이 FTM 에서 Transfer cell 을 구성한 원리를 설명했지만, 논리적이지 않아 삭제했다. 

- FTM, ID injection 을 비교했을 때 어떤 방식이 확실히 좋다고 말하기 애매하다. 

- StyleGAN 을 사용해 해상도를 높였다는 점은 의미 있으나, 성능이 개선되어져야 할 필요는 있다. 
