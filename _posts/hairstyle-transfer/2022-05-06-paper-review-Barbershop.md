---
layout: post
title:  "[Paper Review] Barbershop: GAN-based Image Compusiting using Segmentation Masks"
author: 정정영
categories: [Deep Learning, Hairstyle Transfer, StyleGAN]
image: assets/images/logo-hairstyle-transfer.jpeg
---

![Author](/assets/posts/hairstyle-transfer/barbershop/1.author.png)

![result_grid_image](/assets/posts/hairstyle-transfer/barbershop/2.grid.png)

# 1. Key Idea
- 합성할 이미지들의 segmentation mask 를 통일하면 합성했을 때 artifact 가 작다.

# 2. Backgrounds
- Face editing 은 latent space manipulation 과 image composition 등 두 가지 방법이 있다.

- image composition 은 N 장의 reference image 에서 필요한 영역을 가져와 합치는 방법이다.

- image composition 은 난이도가 높은데, 다음과 같은 어려움들이 있다. 
    1) 이미지 속 요소들이 서로 독립적이지 않다는 점  
    2) hair color 는 조명, 반사광, 주변 색상에 영향을 받는다는 점  
    3) hair style 이 바뀌면서 가려져 있던 요소들(귀, 이마, 턱선 등)이 드러난다는 점  
    4) hair shape 은 구도에 따라 다르게 보일 수 있다는 점  

- 이 논문에서는 이미지 한 장에 대해 다음과 같은 4가지 속성을 정의한다.
    1) apperance: fine details (such as hair colors)  
    2) structure: coarser features (such as form of locks of hair)   
    3) shape: binary segmentation regions  
    4) identity: all the features required to identify an individual  
   
- 서로 다른 이미지에서 위 속성을 하나씩 가져올 때, 최대 4장의 이미지를 하나로 합칠 수 있다. 

# 3. Methodology
## 3-1. Overview
- Reference Image 들로부터 Semantic regions 을 선택해 이미지를 합성한다. 

- 이들을 자동으로 segmentation 한 뒤, target segmentation mask M을 만든다.

- Reference image $I_1$, $I_2$, ... , $I_k$ 는 모두 M 에 대해 align 하고, blending 한다.

- 제안하는 방법은 blending region 의 boundary artifact 를 줄일 수 있다. 

- StyleGAN2 architecture 와 II2S embedding algorithm 을 사용했다. 

![FS_latent_space](/assets/posts/hairstyle-transfer/barbershop/3.FS_latent_space.png)

- Latent code $C(F, S)$ 는 $F\in R^{32 \times 32 \times 512}$ 와 $S\in R^{11 \times 512}$ 를 포함한다. 

  - 이는 기존의 $W^{18 \times 512}$ 보다 자유도가 높아 더 많은 디테일을 표현할 수 있다. 
 
  - 하지만 latent code manipulation 을 하면 artifact 가 발생하기 쉽다. 
  
- 각 step마다 optimizer을 이용한 최적화 과정이 존재한다.

## 3-2. Initial segmentation

- 서로 다른 reference image 에서 합성하려는 영역을 모아놓은 target segmentation mask M 을 만들기 위해 

- 픽셀 $(x, y)$ 에 대해 $M_k(x, y) = k$ 를 만족하는 $k$ 가 있으면 $M(x, y) = k$ 로 설정하고, 이를 만족하는 $k$ 가 2개 이상이면 높은 값을 가져온다. 

- 만약 이를 만족하는 $k$ 가 없다면 heuristic method 로 inpainting 해야 한다. 

> 
- 자동화를 지향하지만, semantic region 을 수동으로 편집해서 scale/translate 을 조절하면 오류를 줄일 수 있다.  
- 특히 reference images' pose 가 서로 다를 경우 오류를 줄일 수 있다.  

![Whole_step](/assets/posts/hairstyle-transfer/barbershop/4.Whole_step.png)

## 3-3. Embedding

- 이미지를 합치기 전에, 각 이미지들을 M에 맞춰 align 해야 한다. 
- 다음과 같은 두 파트로 나뉜다.  
	1) $I_k$ 를 표현하는 $C_k^{rec}$ 를 찾는 reconstruction 과   
    2) $C_k^{rec}$ 의 근처에서 M 에 대응되는 $C_k^{align}$ 을 찾는 alignment    

### 3-3-1. Reconstruction

- $I_k = G(C_k^{rec})$ 를 만족하는 $C_k^{rec}$ 을 찾는다. 

- W+ space 는 W space 에 비해 표현력이 좋지만, 여전히 충분하진 않다. 

- 우리는 FS Space 를 사용해 W+ space 보다 더 표현력이 좋다. 

- FS space 는 spatially correlated structure tensor F 를 포함하고 있어 detail 을 잘 잡는다.



> - W+ space embedding 에서 noise embedding 까지 사용하면 원래 이미지를 거의 완벽하게 복원할 수 있지만, overfitting 이 심해 downstream task (editing / compositing) 에서 artifact 가 심하다고 한다.


### 3-3-2. Alignment

- Reconstruction 단계를 통해 $I_k$ 는 $C_k^{rec} = (F_k^{rec}, S_k^{rec})$ 로 표현된다. 

- $C_k^{rec}$ 는 $I_k$ 를 잘 복원할 수 있지만, M 과는 align 이 맞지 않는다. 

- 그래서 $C_k^{rec}$ 의 근처에 있으면서 M 과 align 이 맞아 있는 $C_k^{align}$ 을 찾는다.

- $F_k^{rec}$ 은 spatially correlated 되어 있기 때문에 $C_k^{align}$ 을 바로 찾는건 어렵다. 

- 대신 aligned image 의 W+ latent vector 인  $w_k^{align}$ 을 찾은 뒤 이를 이용해 $F_k^{rec}$ 에서 $F_k^{align}$ 로 옮겨간다. 

- Semantic segmentation 함수 $segment$ 와 $G$ 를 합성한 $Seg \circ G$ 는 미분 가능한 sementic segmentation generator 이다. 

- Target segmentation mask M 을 GAN inversion 하면 $w_k^{align}$ 을 찾을 수 있다. 

$$
L_{align} = MSE(M, segment(G(w)))
$$

- 하지만 segmentation mask M 에 대응하는 이미지는 많을 수 있으므로, appearance reference image 와 비슷한 이미지를 찾으려면 style loss 를 추가한다. 

$$
L_{align} = MSE(M, segment(G(w))) + \lambda_sL_s
$$


> 
- XENT 는 cross entropy loss function 을 의미한다. 
- $F_k^{align}$ 를 찾는게 목적이기 때문에, 이 과정에서 F 에 해당하는 w+ latent code 만 optimize 했다. 
- $w_k^{align}$ 가 $w_k^{rec}$ 근처에 있어야 하기 때문에 early stopping 했고, 100 iteration 에서 멈췄으나 50-200 iteration 의 성능은 비슷했다. 
- Style loss, L1, L2 를 다 써봤는데, Style loss 만 쓰는게 낫다. Style loss 는 Gram matrix 를 사용했다. 
- mask 와 image 모두 iteration step 마다 update 된다.

### 3-3-3. Structure Transfer
- reconstructed image에 hair structure을 transfer하는 과정이다.
-  target image의 F latent vector 중 target mask의 머리에 해당하는 영역만 reference image의 F latent vector로 교체한다. 

- binary mask를 이용하여 structure를 가져온다.  

$$
\alpha_{k}(x,y) = 1*{M(x,y) = k}
$$

$$
\beta _{k}(x,y) = 1*{M _{k}(x,y) = k}
$$

>
- $\alpha$는 target mask M에서 structure transfer할 얼굴 영역을 선택한 binary mask이다.
- $\beta$는 reference image's mask $M_{k}$에서 structure transfer할 얼굴 영역을 선택한 binary mask이다.

$$
F^{align}_{k} = \alpha_{k,m}*\beta_{k,m}*F^{rec}_{k} + (1-\alpha_{k,m}*\beta_{k,m})*G_{m}(w^{align}_{k})
$$

>
- $G_{m}(w^{align}_{k})$ : Generator안 m번째 style-block의 output

- structure transfer하는 영역은 target mask M과 reference mask $M_{k}$의 바꿀 얼굴 영역이 겹치는 부분으로 한정된다.

## 3-4. Structure Blending
- blended image를 생성하기 위해 $C^{align} _{k}$의 structure tensor($F^{align} _{k}$)를 섞는 방식으로 진행한다.  
- 각 structure tensor들의 영역을 weights($\alpha_{k,m}$)만큼 섞어준다.


$$
F^{blend} = \sum _{k=1}^{K}\alpha _{k,m}\bigodot F^{align} _{k}
$$

## 3-5. Appearance Blending
- reference codes $S_{k}(k=1,...,K)$를 조합하여 style code $S^{blend}$를 찾는 방식이다.

- 먼저 각 reference image의 style code$S_{k}(k=1,...,K)$를 찾기 위해 optimize 한다. optimize할 때 loss는 *masked version of LPIPS*를 이용하여 최적화를 진행하였다.
- 기존 LPIPS를 구하는 식에서 loss를 계산할 영역을 정한 mask($\alpha_{k,l}$)를 추가하였다. 

$$
L^{mask}=\sum _{kl}\alpha _{k,l}L _{PIPS}
$$


- k개 style code를 얻었으면 Blending하여style code($S^{blend}$)로 만들어야 한다.  

$$
S^{blend} = \sum _{k}u _{k}\bigodot S _{k}
$$
  
$$
\sum_{k}u _{k} = 1 (u _{k} \geq 0)
$$

- Structure blending과 비슷하게 weight($u_{k}$)를 이용하여 $S_{K}$를 섞어준다.


# 4. Results
- image를 embedding 하는데 걸리는 시간
	- W latent space : 2 minutes per image
    - C latent space : 1 minutes per image
- image를 align 하는데 걸리는 시간
	- 2 minutes per image
- image를 blending 하는데 걸리는 시간
	- 2 minutes per image

![result_grid](/assets/posts/hairstyle-transfer/barbershop/5.result_grid.png)

## 4.1 Comparison
- 비교 모델로 MichiGAN, LOHO 를 선택하였다.

### User Study
![compare](/assets/posts/hairstyle-transfer/barbershop/6.compare.png)

- 이미지의 디테일 퀄리티, identity의 유지가 잘 이루어 졌는지 관련하여 조사했다.
- 약 96% 정도의 설문 참가자들이 barbershop의 결과가 더 낫다고 평가하였다.

### Reconstruction Quality
![recon_study](/assets/posts/hairstyle-transfer/barbershop/7.recon_study.png)

## 4.2 Ablation Study
![ablation_study](/assets/posts/hairstyle-transfer/barbershop/8.ablation_study.png)

- w/o Align
Align 하지 않아서 머리와 얼굴이 잘 조합되지 않는다. 또한 머리 영역 가장자리에 아티펙트가 보인다.

- using $F^{rec} _{k}$ rather then $F^{align} _{k}$ when blending
original image의 detail한 부분까지 재현 가능하였다. 그러나 semantic alignment 부분이 부족하였다.

- W+ vs FS latent space
FS latent space가 identity 유지에 더 용이하다.

## 4.5 Qualitative Results
### edit semantic region
barbershop에서는 hair를 중심으로 appearance, structure transfer를 진행하였다.

![other_usage](/assets/posts/hairstyle-transfer/barbershop/9.other_usage.png)

추가적으로 Fig. 7을 살펴보면 머리 외에도 다른 얼굴 요소도 style transfer가 가능함을 관찰할 수 있다.

**Improvements**
1. Barbershop 처럼 가져올 영역을 그대로 복사하여 붙여넣기 하는게 이미지 생성의 퀄리티를 높아짐을 보였다.
2. 다른 방법들은 misalignment 영역이 존재하면 쉽게 artifact가 발생한다.
3. 또한 다른 방법들은 머리 영역과 머리 외 영역 간 빛 정보가 다르다면 생성되는 이미지 퀄리티가 낮아지는 모습을 보인다.

## 4.6 Limitation

![limitation](/assets/posts/hairstyle-transfer/barbershop/10.limitation.png)

- 여전히 보석 reconstruct가 잘 안된다.
- 얼굴 위를 가리는 얇은 머리카락같은 경우 생성하기 힘들다.
- aligning을 진행할 때 M영역과 reference의 얼굴 영역이 다른 경우 reference 머리를 표현하지 못한다.(이 경우 머릿결이 표현되지 않고 smoother structure 모습을 보인다.)