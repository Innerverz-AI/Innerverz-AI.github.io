---
layout: post
title:  "[Paper Review] SPADE : Semantic Image Synthesis with Spatially-Adaptive Normalization"
author: 정정영
categories: [Deep Learning, Spatially-Adaptive Normalization, SPADE, face-design]
image: 
---

![Author](/assets/posts/face-design/SPADE/1.author.png)

![result_grid_image](/assets/posts/face-design/SPADE/2.result_gird_image.png)

# 1. Contributions
- Semantic segmentation mask를 photorealistic image로 변환하는 conditional image synthesis의 새로운 방법을 소개한다. 
- Semantic Image를 input으로 넣지 않고 normalization layer이후 modulation을 적용하기 위해서 사용한다.

# 2. Backgrounds
- Conventional network architecture는 convolution, normalization, nonlinearity network가 쌓여있는 구조로 되어있다.
q