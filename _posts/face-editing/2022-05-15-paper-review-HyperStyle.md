---
layout: post
title:  "HyperStyle: StyleGAN Inversion with HyperNetworks for Real Image Editing"
author: 정정영
categories: [Deep Learning, Image editing, StyleGAN, Generator tuning]
image: assets/images/logo-face-editing.jpeg
---
![Author](/assets/posts/hairstyle-transfer/hyperstyle/1.authors.png){: width="600" height="100"}

# 1. Contributions
- optmization + encoder 방식을 이용하여 real-time inference를 수행한다.
- generator(StyleGAN synthesis network)의 parameter를 선택적으로 fine tuning 하는 방법을 소개한다.

# 2. Image Inversion & editing
## Image Inversion
Image를 latent vector로 **잘** embedding하기 위해서 여러 연구가 있었다. 대표적으로  
1. encoder를 이용하여 Image를 latent vector로 embedding 하는 방식(pSp, e4e)  
2. Generator를 fine tuning 하는 방식(PTI)  
3. Input image와 result image를 비교하여 latent vector를 업데이트 하는 방식(optimizer)  

들이 존재한다. 여기서 **잘** embedding 한다는 것은 embedding된 latent vector를 가지고 reconstruct 시 input image와 유사하게 나온다는 의미이다. 

## Image editing
embedding을 통하여 얻은 latent vector를 가지고 input image에 다른 속성을 부여하는 방법 또한 많이 연구되어 왔다. latent vector에 semantic vector를 더하여 수정하는 방식이 있다.(styleflow, sefa, interfaceGAN...) 여기서 **잘** editing하는 의미는 input image의 identity를 유지하면서 원하는 속성을 부여한다는 것을 의미한다.

# 3. GAN Inversion trade-off
## distortion - editability, perceptual quality
해당 부분은 앞 post에서 언급한 바가 있으니 넘어가도록 한다.  
[e4e](https://innerverz-ai.github.io/review-E4E/)

## time-accuracy trade off
위에서 언급한 Image inversion시 시간이 많이 걸릴 수록 reconstruction 정확도가 높아진다는 의미이다.  
optimizer를 이용하는 방식이 시간이 가장 오래 걸리나 정확도가 높으며, encoder를 이용하는 방식은 시간이 가장 적게 걸리나 real image와 reconstruction된 이미지 모습이 많이 차이난다.  

# 4. HyperStyle
encoder를 이용하여 latent vector를 단시간에 얻을 수 있는 장점을 가진 inversion 방식과 encoder 보다 시간은 걸리지만 accuracy가 좋은 generator fine tuning 하는 방식을 혼합하였다.  

## Scheme
![scheme](/assets/posts/hairstyle-transfer/hyperstyle/2.scheme.png){: width="100%" height="80%"}

## Overview
### begin
- x : input image   
- G : Generator
- $\theta$ : generator's parameter
- $ \hat{w}_{init} $ : E(x) (using off-the-shelf encoder, e4e)  
- ![hat_y](/assets/posts/hairstyle-transfer/hyperstyle/hat_y.svg)

### Step
HyperStyle은 input image의 identity를 유지하기 위해서 generator를 fine tuning하는 것이 main idea다. 
1. x, $ G(\theta), \hat{w}_{init} $ 을 준비한다.  
2. Figure 2의 Pre-trained Generator( $G(\theta)$ )에 $ \hat{w}_{init} $ 을 입력하여 $ \hat{y} $ 을 얻는다.
3. x와 $ \hat{y} $ 를 비교하여 loss를 계산한다.
4. Hypernetwork(H)가 offset(= $\Delta$)를 계산하고 generator를 modified 한다.
5. Modified Generator($G(\hat{\theta})$)를 가지고 2~4 번을 반복한다.
6. fine tuning된 Generator를 얻는다.  

( $ \hat{w}_{init} $ 를 구하는 과정은 생략한다.)  
generator's parameter를 수정할 offset을 생성하는 H를 자세히 살펴본다.  

## Hypernetwork(H)
input은 6-channel input (x, $\hat{y}_{init}$ )이나 (x, $ \hat{y} $) 을 받고 output으로 offset(= $\Delta$)을 내보낸다.

H의 구성인 ResNet Backbone과 Refinement Block을 통해 generator tuning할 offset이 나오게 된다.  

저자들은 weight tuning시 다음과 같은 식으로 5번 진행한다.(Restyle에서 소개된 방식)    

![weight_tuning](/assets/posts/hairstyle-transfer/hyperstyle/weight_tuning.svg)

논문에는 StyleGAN2를 generator로 선택하였는데 3.07B에 달하는 generator parameter의 offset을 계산하기에는 시간이 많이 소모된다. 이를 해결하기 위해 저자들은 다음과 같은 전략을 사용하였다.


### 1) Channel-wise로 offset
6-channel input 이 ResNet Backbone으로 들어가게 되면 16x16x512 feature map이 나오고, 이를 Refinement Block에 입력하면 업데이트 할 offset 값이 나오게 된다.

이 offset 값들은 single generator layer의 Conv layer weight를 modulation해줄 값이다. 픽셀별로 Conv layer weight를 학습하기에는 parameter 수가 너무 많으므로 channel-wise로 modulation 한다.

l번째 레이어 parameters $ \theta_{l} $ 에 offset을 적용한다고 가정한다.  
원래 l번째 레이어에는 $ k _{l} * k _{l} * C^{in} _{l} * C^{out} _{l} $ 사이즈의 filter parameters가 존재한다.  

그러나 Channel-wise offset 전략을 이용하면 업데이트 할 parameters 수는 $ 1 * 1 * C^{in} _{l} * C^{out} _{l} $ 로 줄어들게 된다.  

#### Refinement block
ResNet output인 16x16x512 feature map에서 Convolution을 통해 1x1x512 size의 feature map으로 만들고 Full-Connected layer를 이용하여 $1 * 1 * C^{in}_{l} * C^{out} _{l} $ size의 offset 값을 얻어낸다.

![refinement_block](/assets/posts/hairstyle-transfer/hyperstyle/8.compare_parameters.png){: width="50%" height="25%"}

이 전략을 통해서 offset의 수는 367M(약 89% 감소)가 되었다. 그러나 본 논문의 목표인 real-time inference를 위해서는 아직 offset parameter가 많다.(Table1을 살펴보면 encoder 방식의 parameter 수에 비해 여전히 많다.)

### 2) Sharing refinement block
StyleGAN의 layer를 기준으로 Conv layer's weight 수가 동일한 부분이 존재한다. 3 x 3 x 512 x 512 size filer demension을 가진 layer들의 offset을 계산하는 refinement block에서 FC layer를 공유하여 학습했다. 그리고 이 block을 sharing refinement block이라 정했다.

> 구체적인 layer는 appendix를 참고바란다.

### 3) Selective layer refinement
- toRGB layer의 weight를 변경하지 않는다.
- StyleGAN generator에서 Medium, Fine 영역 Conv layer만 fine tuning 한다.

## Loss
x와 $ \hat{y} $ 를 비교한다. 식은 다음과 같다.  
$ L_{2}(x,\hat{y}) + \lambda_{LPIPS}L_{LPIPS} + \lambda_{sim}L_{sim}(x,\hat{y}) $

> $L_{sim}$ : arcface를 가지고 비교하였다.

# 4. Experiments
## Datasets(in face domain) & baseline
### Datasets
- FFHQ(for train)
- CeleA-HQ(for test)

### baseline
compared with
- pSp(encoder type)
- e4e(encoder type)
- ReStyle(encoder type)
- PTI(optimization type)

## 4.1. Reconstruction Quality
### Qualitative Evaluation
![reconstruction_quality](/assets/posts/hairstyle-transfer/hyperstyle/4.reconstruction_quality.png){: width="70%" height="50%"}
- HyperStyle 방식이 Optimization 방식 퀄리티랑 비슷하다.
- 다른 방식은 디테일한 표현이 부족하거나 Input image의 identity 보존이 부족하다.

### Quantitative Evaluation
![Quantitative_Evaluation](/assets/posts/hairstyle-transfer/hyperstyle/5.Quantitative_Evaluation.png){: width="50%" height="25%"}

- 저자들은 time-accuracy trade-off를 중심으로 평가하였다.
- 이미지 퀄리티를 평가하기 위해서 $L_{2}$, LPIPS, MS-SSIM 을 이용하였다.
- 생성된 이미지의 ID 보존도를 평가하기 위해서 Curricularface를 이용하였다.

> HyperStyle은 optimization 방식임에도 불구하고 이미지 생성시간이 적게 든다. 또한 짧은 생성 시간에 높은 이미지 퀄리티를 기록한다는 점 또한 눈에 띈다.

## 4.2. Editability via Latent Space Manipulations
### Qualitative Evaluation 
![Editing_quality_comparsion](/assets/posts/hairstyle-transfer/hyperstyle/6.Editing_quality_comparsion.png){: width="70%" height="40%"}

- W+(optimization, pSp, $ReStyle_{pSp}$) : 에서는 edit가 잘 되지 않음. 또한 reconstruction-editability
trade-off가 발생하였다.
- W(PTI, HyperStyle) : W+ latent space 이용 방법보다 좋은 edit 수행 능력을 보여주었다.

### Quantitative Evaluation
![Quantitative_editing_metrics](/assets/posts/hairstyle-transfer/hyperstyle/7.Quantitative_editing_metrics.png){: width="50%" height="25%"}

- 얼굴의 각도 변화와 미소 정도를 조절하여 원본 identity를 잘 유지하고 있는지 측정하였다.
- identity 측정은 Curricularface를 이용하였다.

# 5. Conclusion
- encoder + generator tuning 방식을 혼합하여 이미지 생성 퀄리티와 시간면에서 좋은 결과물을 보였다.
- 시간 문제를 해결하기 위해 generator를 선택적으로 tuning하는 아이디어가 참신했다.