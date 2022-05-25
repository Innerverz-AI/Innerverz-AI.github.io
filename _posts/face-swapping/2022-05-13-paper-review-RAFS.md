---
layout: post
title:  "[Paper Review] RAFS: Region-Aware Face Swapping"
author: 류원종
categories: [Deep Learning, Face Swapping, StyleGAN, Deep Fake]
image: assets/images/logo-face-swapping.jpeg
---

![title](/assets/posts/face-swapping/RAFS/title.PNG){: width="100%", height="100%"}<br>

# 1. Motivation

Face swapping 에서 많이 쓰이고 있는 AdaIN 은 global function 이라 source identity 를 제대로 옮기지 못하는데?

눈, 코, 입술, 눈썹 등 identity-relevant local regions 를 집중적으로 파헤쳐보자! 

# 2. Overview

![overview](/assets/posts/face-swapping/RAFS/overview.PNG){: width="100%", height="100%"}<br>

우선 source 와 target 을 pSp encoder $\phi$ 에 통과시켜 $F_s$ 와 $F_t$ 를 얻고, 동시에 face parsing 까지 한다.
중간 결과물들은 화살표를 따라 처리되는데, 파란색 화살표는 Source Feature-Adaptive (SFA) branch 로, source 의 global feauture 를 처리하는 과정이다. 
주황색 화살표는 Facial Region-Aware (FRA) branch 로, 얼굴의 local feature 를 처리하는 과정이다.
$F_s$ 는 파란색 SFA 를 따라가 source 의 global feature 를 추출하고, 주황색 FRA 를 따라가 local feature 를 뽑아낸다. 
$F_s$ 는 Region-Aware identity Tokenizer (RAT) 를 통과하는데, 그 결과물로 $T_s$ 를 얻는다.
그리고나서 Transformer 를 통과해 $\hat{T_s}$ 를 얻는다. $\hat{T_s}$ 는 Region-Aware identity Projector (RAP) 에서 $F_t$ 에 합쳐진다.
RAP 를 통과하고 나온 $F^l$ 은 $F^g$ 와 만나 $\hat{T_s}$ 가 되고, 이는 StyleGAN 에 들어가 $\hat{I_{s \to t}}$ 를 만들어낸다. 
최종적으로, FMP 모듈을 통해 얻어진 face mask 를 이용해 $\hat{I_{out}}$ 을 얻어낸다.

용어 정리
- RAT: Region-Aware identity Tokenizer
- RAP: Region-Aware identity Projector
- GAP: Global Average Pooling
- FRA: Facial Region-Aware
- SFA: Source Feature-Adaptive


# 3. Methods

## 3.1 Facial Region-Aware Branch
### 3.1.1 Region-Aware Identity Tokenizer (RAT)

![RAT](/assets/posts/face-swapping/RAFS/RAT.PNG){: width="100%", height="100%"}<br>

얼굴 속에서 identity 와 관련 있는 부분만을 처리하기 위해, 눈, 눈썹, 입술, 코 영역의 정보를 token $T_s \in R^{N \times L \times 512}$ 으로 만든다.
여기서 N 은 특징맵의 스케일 수, L 은 영역의 수인데, 이 논문에서는 각각 3과 4로 정해져있다. 
pSp 에서 얻은 3개의 특징맵은 스케일이 서로 다르기 때문에 bilinear interpolation 을 이용해 통일한다. 
또한 4개의 얼굴 요소 영역을 개별적으로 처리하기 위해 region-wise average pooling layer $\Phi$ 를 적용했다. 

$$
T_s^n = Linear(\Phi(F_s, M_s^n)) \\
where\ M_s^n \in \{M_s^{lips}, M_s^{nose}, M_s^{brows}, M_s^{eyes}\}
$$

### 3.1.2 Transformer layers

AdaIN 기반의 방법들은 local feature 를 제대로 다루지 못해 identity 유지력이 떨어졌다. 

### 3.1.3 Region-Aware Identity Projector

![RAP](/assets/posts/face-swapping/RAFS/RAP.PNG){: width="100%", height="100%"}<br>

source token 을 target feature map 에 반영하기 위해 Projector 가 필요하다.
이는 표정이나 눈동자 위치 등 source 와 target 의 misaligned attributes 들을 고려할 수 있어야 한다. 
위 그림에서 볼 수 있듯, 
Region-Aware Identity Project 는 source 의 토큰을 target 에 반영한다. 


## 3.2 Source Feature-Adaptive Branch

FRA 에서 source identity 에 관련된 local feauture 들은 target face 로 잘 합쳐졌지만, 주름, 수염 등 global feature 또한 고려해야 한다. source 와 target 의 misalignment 를 피하기 위해, source 에서 얻은 특징맵 중 가장 작은 것을 global averaging pooling (GAP) 에 통과시킨다. MLP 까지 통과하고 나면 세 가지 크기의 global feature 로 나뉘어 $F_l$ 에 더해진다. 


$$
F^g = MLPs(GAP(F_s^0))\\
\hat{F^t} = F^g + F^l
$$

## 3.3 Face Mask Predictor

![maskpredictor](/assets/posts/face-swapping/RAFS/maskpredictor.PNG){: width="100%", height="100%"}<br>


occlusion, distorted background 등의 문제를 해결하기 위해 soft face masks 를 이용한다. 
stylegan2 의 feature map 을 이용하는데, 크기가 16 인 특징맵부터 256 인 특징맵까지 가져와 32x256x256 크기로 통일한 뒤 이어붙인다. 
그 뒤 1x1 conv 와 sigmoid 함수를 이용해 soft mask M 을 만든다. 

$$
\hat{I}_{out} = M \odot \hat{I}_{s \to t} + (1-M) \odot I_t
$$

# 4. Objective

ArcFace 를 이용한 id loss, pretrained vgg model 을 이용한 perceptual loss, source 와 target 이 같을 경우 L2 loss 를 적용한다.

$$
L_{total} = 0.15 \times L_{id} + L_{rec} + 0.8 \times L_{LPIPS} 
$$

# 5. Implementation Details 


Dataset 은 CelebA-HQ 와 FaceForensics++ 를 사용했다. source 와 target 이 같은 비율은 20% 이고, input size 는 256 으로 설정했다. 
Adam optimizer($\beta1=0.9, \beta2=0.999$)를 사용했으며, lr 은 1e-4 로 설정했다. 1-way Tesla V100 GPU 을 이용해 batch size 8 로 50K step 만큼 학습했다.

# 6. Results

## 6.1 Comparison with existing methods


![result1](/assets/posts/face-swapping/RAFS/result1.PNG){: width="100%", height="100%"}<br>

MegaFs 는 얼굴색도 못 맞추는 반면, 우리는 눈동자 색, 작은 입 등 source 의 특징을 잘 옮겼다. 또한 안경, 머리 색과 같은 target 의 특징도 잘 살린다. 

![result3](/assets/posts/face-swapping/RAFS/result3.PNG){: width="100%", height="100%"}<br>

gender, age, skin color, pose 등에서 gap 이 큰 이미지들을 골라 face swapping 한 결과에서도 이 논문의 성능이 돋보인다. 
특히 눈 색깔, 수염, 주름 등 정보를 잘 가져온다. 

## 6.2 ablation study


### 6.2.1 Feature Fusion module

![result4](/assets/posts/face-swapping/RAFS/result4.PNG){: width="100%", height="100%"}<br>

FRA 와 SFA 의 조합이 AdaIN 에 비해 낫다는 것을 보이기 위해, AdaIN 을 baseline 으로 두고 실험을 진행했다. 
위 이미지에서 볼 수 있듯 AdaIN 은 source identity 를 유지하지 못한다. 
SFA 만을 사용했을 때도 identity 를 잘 보존하지 못하고, FRA 만을 사용했을 때는 identity 는 잘 보존하지만 face texture 에서 오류가 있다. 
SFA 와 FRA 를 모두 사용한 경우 local identity-relevant features 와 global facial detail 을 동시에 유지할 수 있다. 

### 6.2.2 Attention Structure

token interaction 이 가능한 Transformer 의 능력을 검증하기 위해, Transformer layer 을 Non-local layer 로 대체하였다. 


### 6.2.3 Face Mask Predictor

![FMP](/assets/posts/face-swapping/RAFS/FMP.PNG){: width="100%", height="100%"}<br>

face mask 가 없으면 앞머리나 배경이 달라진다. Face Parsing 을 이용해 mask 를 구한 경우 정확하지 않거나 자연스럽지 않은 결과를 보인다. 여기서 제안한 방법은 조화로운 결과를 보인다. 

### 6.2.4 Expanded Application of FRA


![mt](/assets/posts/face-swapping/RAFS/mt.PNG){: width="100%", height="100%"}<br>

FRA branch 를 makeup transfer 에도 적용해보았는데, PSGAN 을 baseline 으로 삼고 AMM module 을 FRA branch 로 교체했다. 
더 자연스러운 결과를 얻을 수 있었고, 속도도 10배 정도 더 빨랐다. 
FRA 가 texture and color feature transfer 를 잘 처리할 수 있음을 보였다. 
이는 token mechanism 의 유연성과 충분한 feature interaction 덕분이다.  


# 7. Conclusion

이 논문에서는 identity-consistent face swapping 이 가능한 방법을 제안했다. FRA 모듈은 identity-relevant local feature 을 target face 에 합쳤고, SFA 모듈은 identity relevant global details 를 보완했다. 뿐만 아니라 unsupervised 방식의 FMP 덕분에 inner-face 와 배경을 자연스럽고 정확하게 합성할 수 있었다.  

# 7.1 Limitation

Training dataset 의 한계로, out-range case 들을 처리하는데 실패했다. 

# 7.2 My opinion
