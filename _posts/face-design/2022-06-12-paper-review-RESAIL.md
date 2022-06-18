---
layout: post
title:  "[Paper Review] RESAIL : Retrieval-based Spatially Adaptive Normalization for Semantic Image Synthesis"
author: 정정영
categories: [Deep Learning, Spatially-Adaptive Normalization, Retrieval-based, face-design]
image: assets/images/logos-face-design2.png
---
![Author](/assets/posts/face-design/RESAIL/1.author.png)

![result_gird](/assets/posts/face-design/RESAIL/2.result_gird.png){: width="100%", height="100%"}<br>  


# 1. Contributions
이때까지 spatially-adaptive normalization은 semantic class의 coarse-level 정보를 받아 image synthesis를 수행하였다.(CLADE, OASIS) semantic label에 일정한 패턴이 있는 object인 경우(차 처럼 바퀴, 창문이 일정한 위치에 존재하는 경우) semantic image만으로 이를 파악하지 못하고 blur한 이미지를 생성하는 limitation이 존재한다. 그래서 본 논문에는 retrieval-based spatially adaptive normalization으로 fine detail을 살리는 방법을 제시한다.

- semantic class 모습과 비슷한 content patch를 dataset에서 가져와 find-grained modulation을 수행한다.
- distorted content patch로 만든 guidance image를 가지고 model을 학습하였다.

# 2. Method
![result_gird](/assets/posts/face-design/RESAIL/3.method.png){: width="100%", height="100%"}<br>  
전체적인 과정을 살펴보자면,
1. semantic map을 retrieval paradigm로 보내 guidance image $I^{r}$ 을 얻어낸다.
2. guidance image과 semantic map을 Retrieval-based Spatially Adaptive Normalization(RESAIL) 을 이용한 generator에 통과시켜 최종 이미지 $\hat{I}$ 를 얻는다.

## 2.1. Retrieval-based Guidance
retrieval function(retrieval paradigm) 안에서 어떤 과정을 거치는지 알아본다.(Fig. 2(a) 참조)  

1. Semantic map M을 object별로 분해하여 mask를 만든다.  

    ![M](/assets/posts/face-design/RESAIL/M.svg)

    > - $M^{s}_{i}$ : 한 object를 crop 한 binary segment mask  
    > - $y^{c}_{i}$ : M mask에 대응하는 category

2. training images는 1번 과정 처럼 semantic map 정보를 이용하여 object들을 training image에서 분리한다.  
(분리된 object들은 retrieval unit이라 한다.)
3. semantic map M에서 나온 object별 mask와 training image에서 나온 object별 mask가 비슷하게 일치하는 경우 Retrieved Segments로 정한다.
4. Retriebed Segments들을 재조합하여 guidance image $I^{r}$ 를 얻는다.

retrieval-based guidance($I^{r}$)는 아래 식으로 얻어진다.  

![guidance_image](/assets/posts/face-design/RESAIL/guidance_image.svg)


> - Γ(D,M,y): training dataset에서 진행하는 retrieval function을 의미.
> - $M^{s}_{i}$ : mask와 가장 비슷한 모습을 가진 object의 segment image  
(만약 비슷한 object가 없으면 검은색으로 칠한다. 또한 빈 영역이 발생하면 그 부분을 검은색으로 칠한다.)
> - $\Theta$ : dataset에서 찾은 object의 segment image들을 재조합
> - 학습할 경우 dataset에서 training image를 빼고 진행한다. 

## 2.2. Distorted Ground-truth as Guidance
supervision 방식의 학습을 위해서는 pair가 되는 ground trues image와 guidance image가 있어야 한다. 그래야 guidance image를 가지고 ground trues image를 잘 생성했는지 loss를 계산할 수 있기 때문이다. 그러나 위 Retrieval-based guidance image는 ground trues image가 존재하지 않아 supervision 방식 학습이 불가하였다.

![distorted_GT](/assets/posts/face-design/RESAIL/7.distorted_GT.png)  
이를 해결하기 위해서 Ground trues image를 객체별로 distortion 뒤에 재결합하여 Distored GT ($\widetilde{I}^{gt}$)를 만든다. 이렇게 하면 pair dataset이 만들어 지므로 supervision 학습이 가능해진다.

## 2.3 Network Architecture
### Retrieval-based Spatially Adaptive Normalization(RE-SAIL)
Conditional normalization architecture을 가진 RE-SAIL은 guidance image $I^{r}$(or $\widetilde{I}^{gt}$)와 semantic map M을 input으로 받고 결과물로 Batch Norm을 거친 feature map에 modulation을 한다.

Semantic Map은 SPADE와 비슷하게 진행하지만, Guidance image는 four-layer convolutional network를 통해 scale($\gamma^{r}$), bias($\beta^{r}$)를 얻는다. Guidance image를 만들면서 발생한 검은영역에 대한 정보를 채워주기 위해 만들어진 구조라 파악된다.(Fig 2.(c) 참조)

#### Step1. 3x3 Conv layer
Guidance image는 object 사이에 검은 영역이 있으므로 3x3 Conv layer를 통해 채워준다.

#### Step2. two AdaIN layer
추가적으로 검은 영역에 semantic information 정보를 넣어주기 위해서 1x1 Conv를 거친 Semantic Map을 Guidance image에 ADAIN을 적용한다.

#### Step3. mix
Semantic Map과 Guidance image를 통해 얻은 $\gamma^{r}$, $\beta^{r}$, $\gamma^{s}$, $\beta^{s}$ 를 learnable weight paramter ($\alpha_{\gamma}$, $\alpha_{\beta}$) 를 가지고 적절하게 섞어준다.
$
\gamma = \alpha_{\gamma}\gamma^{s} + (1-\alpha_{\gamma})\gamma^{r}
$

$
\beta = \alpha_{\beta}\beta^{s} + (1-\alpha_{\beta})\beta^{r}
$

구조 자체는 SEAN과 매우 비슷하게 보인다.

### How to modulate input actiavtions by RE-SAIL 
$
RESAIL(h,M,I^{r}) = \gamma_{c,y,x}\frac{h_{n,c,y,x} - \mu_{c}}{\sigma_{c}} + \beta_{c,y,x}
$
> - h : N개 samples에 대한 input actiavtions
> - $\mu_{c}, \sigma_{c}$ : h의 평균과 표준편차

### Generator
SPADE generator 구조와 거의 흡사하다. upsampling layer를 통과할때 마다 RESAIL ResBlk를 적용하였다.

$ \hat{I} = G(M, I^{r}), \hat{I}^{gt} = G(M, \hat{I}^{gt}) $

> - $\hat{I}^{gt}$ or $I^{r}$ : guidance image
> - M : semantic map 
> - $\hat{I}$ : synthesized image

## 2.4. Loss functions

![loss](/assets/posts/face-design/RESAIL/loss.svg)

> - $\hat{I}^{gt}$, ${I}^{gt}$ 간에 perceptual loss와 feature matching loss를 적용하였다.
> - $\hat{I}^{gt}$, $\hat{I}$ 을 사실적으로 만들기 위해 adversial loss를 걸어주었다.

### $ \mathbb{L}_{cls}$
각 semantic region에 synthesis를 잘 하기 위해 만든 segmentation loss이다. pretrained segmentation network S를 이용하여 생성된 이미지를 semantic image로 만들었다. 그리고 segmentation mask M과 비교하여 loss를 계산한다.  
![cls_loss](/assets/posts/face-design/RESAIL/cls_loss.svg)

> - $\alpha_{c}$ : class balancing weight

# 3. Experiments
## 3.1. Experimental Setting
### Datasets
- Cityscapes(3k images, 35 semantic categories)
- ADE20K(over 20k images, 150 semantic categories)
- ADE20K-outdoor(subset of ADE20K dataset)
- COCO-Stuff(118k images)

### Evaluation Metric
- Pixel ACcuracy(AC)
- mean Intersection-Over_Union(mIOU)
- segmentation accuracy
- FID

## 3.2. Qualitative Results
![qualitative_results](/assets/posts/face-design/RESAIL/4.qualitative_results.png){: width="100%", height="100%"}<br>  

- RESAIL model 결과물이 더 사실적이며, 물체의 구조적 특징도 잘 반영된 결과가 나왔다.

### Multi-modal synthesis capability
![multi_modal_test](/assets/posts/face-design/RESAIL/5.multi_modal_test.png){: width="100%", height="100%"}<br>   

## 3.3. Quantitative Results
![quantitative_result](/assets/posts/face-design/RESAIL/6.quantitative_result.png)   

- 거의 모든 dataset에서 RESAIL이 좋은 결과를 보인다.

## 3.4. Ablation Studies
### Effectiveness of RESAIL Module
![ablation_compare_module](/assets/posts/face-design/RESAIL/8.ablation_compare_module.png)  

### Effectiveness of Distorted Ground-truth
![Effectiveness_of_Distorted_GT](/assets/posts/face-design/RESAIL/9.ablation_effect_distorted_GT.png)  

# 4. Conclusion
- RESAIL module을 사용하여 fine detail한 영역에도 image synthesis가 잘 된다는 것을 증명하였다.
- 그러나 inference speed가 느리다는 한계점이 있다. 
- 특히 Retrieving operation에서 시간을 많이 먹는다.

# 5. Think
- 이때까지 살펴본 논문들은 input image에 대해서 style vector를 뽑아서 semantic region을 채워넣다 보니 object 정보들이 많이 소실된것 같다.
- object에 대한 정보를 이미지 그대로 이용하였기 때문에 잘 나오는게 당연해 보인다.
- 궁금한 점은 oject mask에 대응되는 retrieved segments를 dataset에서 가져오는게 아니라 원하는 물체를 넣을 수 있는지 궁금하다.