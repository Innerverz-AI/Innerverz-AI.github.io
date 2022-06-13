---
layout: post
title:  "[Paper Review] RESAIL : Retrieval-based Spatially Adaptive Normalization for Semantic Image Synthesis"
author: 정정영
categories: [Deep Learning, Spatially-Adaptive Normalization, Retrieval-based, face-design]
image: 
---
![Author](/assets/posts/face-design/RESAIL/1.author.png)

![result_gird](/assets/posts/face-design/RESAIL/2.result_gird.png)


# 1. Contributions
이때까지 spatially-adaptive normalization은 semantic class의 coarse-level 정보를 받아 image synthesis를 수행하였다.(CLADE, OASIS) semantic label에 일정한 패턴이 있는 object인 경우(차 처럼 바퀴, 창문이 일정한 위치에 존재하는 경우) semantic image만으로 이를 파악하지 못하고 blur한 이미지를 생성하는 limitation이 존재한다. 그래서 본 논문에는 retrieval-based spatially adaptive normalization으로 fine detail을 살리는 방법을 제시한다.

- semantic class 모습과 비슷한 content patch를 dataset에서 가져와 find-grained modulation을 수행한다.
- distorted content patch로 만든 guidance image를 가지고 model을 학습하였다.

# 3. Method
![result_gird](/assets/posts/face-design/RESAIL/3.method.png)
먼저 semantic map을 retrieval paradigm로 보내 guidance image $I^{r}$ 을 얻어낸다. 이후 guidance image과 semantic map을 Retrieval-based Spatially Adaptive Normalization(RESAIL) 을 이용한 generator에 통과시켜 최종 이미지 $\hat{I}$ 를 얻는다.

## 3.1. Retrieval-based Guidance
guidance image $I^{r}$ 을 얻는 방법을 소개한다.

1. 