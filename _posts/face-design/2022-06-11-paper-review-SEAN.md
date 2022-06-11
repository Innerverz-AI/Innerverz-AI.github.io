---
layout: post
title:  "[Paper Review] SEAN : Image Synthesis with Semantic Region-Adaptive Normalization"
author: 정정영
categories: [Deep Learning, Spatially-Adaptive Normalization, SEAN, face-design]
image: 
---

![Author](/assets/posts/face-design/SEAN/1.author.png)

![result_gird](/assets/posts/face-design/SEAN/2.result_gird.png)

# 1. Contributions
- 이전에 언급되었던 SPADE와 OASIS와 달리 원하는 스타일을 각 semantic image region에 주는 방법을 제시한다.
- 네트워크에 style 정보를 넣어줄 때 AdaIN과 비슷한 방식을 이용하였다.
- semantic image region에 따라서 style정보를 넣어주는 normalization 방식을 제시한다.

# 2. Previous Works
## SPADE
- semantic region label과 일치하는 이미지를 생성하는 task
- style code를 하나밖에 사용하지 않는다.
- semnatic region 별로 다른 스타일을 부여하지 못한다.
- result image의 quality가 낮다.
## AdaIN
- A image에 B image의 style을 부여하는 normalization 방법
- 항상 이미지 전체에 대해서 normalization한다.
- 영역별로 다른 normalization을 부여하지 못한다.

# 3. Method
## 3.1 Style Encode 
SEAN에서 소개하는 style encoder는 Encoder-decoder와 semantic image region 별로 average pooling하는 형식으로 진행된다.  

input 으로 real image가 들어오면 Style Encoder를 거쳐 512 x s dimensional style matrix **ST**가 output으로 나오게 된다. 각 dimension 마다 대응되는 semantic image region 영역에 대한 style vector 정보를 가지고 있다.  

## 3.2 Style Control
본 논문에서는 style control을 위해서 Semantic Region-Adaptive Normalization(SEAN)을 새롭게 소개한다. SEAN은 생성된 per-region style code와 segmentation mask를 input으로 받아 ADAIN과 비슷하게 modulating을 진행한다.


### SEAN normalization
![SEAN_normalization](/assets/posts/face-design/SEAN/4.SEAN_normalization.png)  
SEAN normalization은 style matrix **ST**와 segmentation mask를 input으로 받고 여러 conv layer를 거쳐 Batch Norm된 input feature map에 modulating을 진행한다.  

- Segmentation mask는 SPADE와 동일한 방식으로 진행되어 modulation parameters $\gamma^{o}$, $\beta^{o}$ 를 얻는다.
- Style Matrix는 style별로 convolution을 진행한 결과에 segmentation mask를 broadcast 한 결과물을 SPADE와 동일한 방식으로 진행하여 modulation parameters $\gamma^{s}$, $\beta^{s}$ 를 얻는다.

각 modulation parameters는 learnable parameter $\alpha$ 를 통하여 적절하게 input feature map에 반영된다.

$$
\gamma_{c,y,x}(ST, M)\frac{h_{n,c,y,x}-\mu_{c}}{\sigma_{c}} + \beta_{c,y,x}(ST, M)
$$
> - h : input feature map
> - $\mu_{c}$, $\sigma_{c}$ : input feature map's channel c의 평균과 표준편차 
> - n : batch / c : channel

$$
\gamma_{c,y,x}(ST, M) = \alpha_{\gamma}\gamma^{s}_{c,y,x}(ST) + (1-\alpha_{\gamma})\gamma^{o}_{c,y,x}(M)
$$
$$
\beta_{c,y,x}(ST, M) = \alpha_{\beta}\beta^{s}_{c,y,x}(ST) + (1-\alpha_{\beta})\beta^{o}_{c,y,x}(M)
$$

# 4. Experiments
- SEAN은 SPADE 구조를 기반으로 만들어 졌다.
- SPADE upsampling layer에 SEAN ResNet block(SEAN ResBlk)를 적용하였다.

## 4.1. SEAN ResBlk
![network](/assets/posts/face-design/SEAN/3.network.png)

SEAN의 generator 구조는 StyleGAN의 영향을 많이 받았다. Style 정보를 Generator process 중간에 넣어주는 방식, style code가 들어가는 layer의 위치에 따라 결정되는 style 요소가 다른 것 또한 비슷하다.(SEAN generator에는 layer별로 다른 style 정보를 넣어준다. 이는 w+을 따라한듯.) 또한 noise(B)를 추가하여 SEAN 생성 능력을 향상시켰다.

추가적으로 SEAN ResBlk 내부에는 SEAN이 3개 들어있는데, 각 SEAN마다 다르게 style matrix가 들어간다. 이 이유는 ablation study를 통해 알 수 있다.

![ablation_fig](/assets/posts/face-design/SEAN/5.ablation_fig.png)

![ablation_fig](/assets/posts/face-design/SEAN/6.ablation_table.png)
> SEAN-level encoder가 전반적으로 더 우수함을 보인다.

# 5. Result
### Dataset
- CelebAMask-HQ
- CityScapes
- ADE20K

### Compare models
- SPADE
- Pix2PixHD

### Metrics
1. segmentation accuracy measured by mean Intersection-over-Union(mIoU)
2. FID
3. PSNR
4. SSIM
5. RMSE

## 5.2. Quantitative comparsions
![quantitative_comparsions1](/assets/posts/face-design/SEAN/7.quantitative_comparsions1.png)

![quantitative_comparsions2](/assets/posts/face-design/SEAN/8.quantitative_comparsions2.png)

## 5.3. Qualitative results
![qualitative_results](/assets/posts/face-design/SEAN/9.qualitative_results.png)
- Pix2PixHD, SPADE는 측면 얼굴과 늙은 사람을 잘 생성하지 못한다.

## 5.4. Ablation studies
![ablation_fig](/assets/posts/face-design/SEAN/6.ablation_table.png)
- Pix2PixHD이 SPADE 보다 더 나은 style extraction subnetwork를 가지고 있음  
(PSNR reconstruction value가 더 높으므로)
- 반면, SPADE는 Pix2PixHD보다 더 나은 generator를 가지고 있음  
(FID 값이 더 낮으므로)
- SEAN에서는 downsmapling / encoder / noise 의 유무 테스트를 통해서 성능이 더 좋음을 보이고 있다.

# 6. Conclusion
- generator는 stylegan base에 style 정보는 SPADE base normalization을 적용하였더니 고화질 이미지를 영역별로 style을 부여하여 생성할 수 있었다.
- SEAN에서 소개한 encoder를 통해 style vector를 가지고 style interpolation이 가능하다.
- face domain에서 안경쓴 모습 같은 특이한 경우는 잘 생성하지 못한다.