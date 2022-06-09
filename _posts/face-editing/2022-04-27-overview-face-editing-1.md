---
layout: post
title:  "[Survey] Face Editing 최신 논문 몰아보기: (1) Image2StyleGAN, Pixel2Style2Pixel"
author: 이유경
categories: [Face Editing, GAN Inversion, StyleGAN]
image: assets/images/logo-face-editing.jpeg
---

우선 **face editing**에 대해 간단히 알아보자. 이 과정은 크게 세 단계로 이루어진다. 
- 입력받은 이미지를 잘 복원할 수 있는 적절한 latent representation을 찾아내는 '**GAN inversion**'
- 원하는 수정 방향에 맞게 latent representation을 조작하는 '**latent manipulation**'
- 조작된 latent representation로부터 새로운 이미지를 만들어내는 '**image generation**'

본 시리즈에서는 face editing에 관련된 state-of-the-art 모델을 시간 순으로 소개해보려고 한다. 아래에 소개할 논문들을 적어놨다. 원종 님께서 짚어주신 핵심 아이디어 위주로 작성할 예정이다.

**GAN inversion**
- Image2StyleGAN
- Pixel2Style2Pixel (a.k.a. pSp)
- Encoder4Editing (a.k.a. e4e)
- Pivotal Tuning Inversion (a.k.a. PTI)

**Image generation**
- StyleGAN
- StyleGAN2
- StyleGAN3


-----

# GAN inversion
## Image2StyleGAN 
- Abdal et al., "Image2StyleGAN: How to Embed Images Into the StyleGAN Latent Space?" (ICCV 2019) [[arXiv](https://arxiv.org/abs/1904.03189)] <br>

StyleGAN은 intermediate latent $W$ space를 도입해서 사실적인 얼굴 이미지를 합성하고, 일부 특징을 수정하는 데 성공했다. 하지만 랜덤하게 합성된 얼굴 이미지를 다루는 건 내가 원하는 얼굴 이미지를 가지고 노는 것보다 재미가 없다. Image2StyleGAN은 **주어진 이미지 한 장을 확장된 latent $W^+$ space의 한 점에 대응시키는 방법**을 제안했다. 결과적으로 우리는 face editing에 원하는 얼굴 이미지를 사용할 수 있게 되었다.

### Latent $W^+$ space
StyleGAN generator에서는 resolution마다 2번씩, 총 18번 AdaIN 연산을 수행한다. 이때, 모든 AdaIN 연산에 512-dimentional vector $w \in W$가 공통적으로 사용된다. 반면, 이미지를 extended latent $W^+$
space로 mapping하는 경우에는 **18개의 서로 다른 512-dimensional $w$ vector**를 AdaIN 입력으로 사용한다. 

### Embedding algorithm
![algorithm](/assets/posts/face-editing/overview-face-editing/image2stylegan-algorithm.png){: width="70%", height="70%"} <br>

특이한 점은 큰 데이터셋을 사용하지 않고 이미지 $I$ 하나만 사용해서 학습을 진행한다. 또한 optimizer $F'$이 업데이트 하는 대상이 StyleGAN generator $G(\cdot)$가 아닌 latent code $w$라는 점에도 주목해야한다. 학습이 끝난 시점에 우리는 $I$에 완벽히 대응되는 $w$를 얻게 된다. 이러한 GAN inversion 방식을 **optimization-based**라고 이야기한다.  

### Results
![result](/assets/posts/face-editing/overview-face-editing/image2stylegan-result.png){: width="80%", height="80%"} <br>

첫번째 행의 이미지를 Image2StyleGAN에 입력해서 latent representation을 추출했다. 두번째 행의 이미지는 이를 StyleGAN AdaIN 연산에 사용해서 복원한 것이다. 특히 extended latent $W^+$
space를 사용했을 때, 원본 얼굴 이미지가 거의 완벽하게 복원되었다. 

## Pixel2Style2Pixel
- Richardson et al., "Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation"  (CVPR 2021) [[arXiv](https://arxiv.org/abs/2008.00951)] [[project page](https://eladrich.github.io/pixel2style2pixel/)]

Image2StyleGAN처럼 per-image optimization을 하게 되면 이미지 한 장에 대한 latent representation을 계산하는데 수 분이 걸린다. Pixel2Style2Pixel은 **어떠한 이미지가 주어지더라도 one-shot으로 $W^+$ space 위의 점에 대응시킬 수 있는 방법**을 제안했다. 

### Architecture
![architecture](/assets/posts/face-editing/overview-face-editing/pixel2style2pixel-architecture.png){: width="90%", height="90%"}<br>

Pixel2Style2Pixel encoder는 입력받은 이미지로부터 세 가지 scale의 feature map (a.k.a. feature pyramid)을 추출한다. `map2style`이라고 하는 작은 fully convolutional network에서 feature map들을 latent style vector로 바꾼다. 각각의 512-dimentional vector는 StyleGAN generator의 정해진 AdaIN 연산으로 전달된다. <br>
이 모델은 데이터셋을 사용해서 모델 파라미터들을 학습시킬 수 있다. 따라서 어떤 이미지에 대해서도 빠른 속도로 $W^+$ mapping이 가능하다. 한 가지 특징은 **bottleneck 때문에 latent style vector가 변화하는 폭에 제약**이 있다. 이 제약으로 인해 입력받은 이미지의 세세한 디테일이 표현되지 않아서 완벽하게 원래 이미지를 복원할 수 없다.  

### Reconstruction vs. Editing
주어진 이미지를 가장 잘 복원하는 방법은 18 x 512-dimensional $w \in W^+$ vector에 최대한 많은 디테일을 담는 것이다. 이 관점에서는 Image2StyleGAN과 같은 per-image optimization 방법들이 아주 강력하다. 
그렇다면 이미지를 내가 원하는대로 잘 수정하는 방법을 무엇일까? 결과 이미지를 합성해내는 StyleGAN generator가 $W^+$가 아닌 $W$ space에서 학습되었다는 점에서 그 단서를 찾을 수 있다. **Encoder에서 얻은 $w \in W^+$ vector가 $W$ space에서 적게 벗어나 있을수록, 즉 $w_1, w_2, \cdots, w_{18}$이 서로 비슷할 때, 이미지가 잘 수정된다.** Pixel2Style2Pixel 방법은 bottleneck으로 인한 제약 덕분에 이미지 수정 성능이 높다.

### Results
![result](/assets/posts/face-editing/overview-face-editing/pixel2style2pixel-result.png){: width="90%", height="90%"} <br>

GAN inversion (왼쪽 위), multi-modal conditioning (오른쪽 위 2세트), face frontalization (왼쪽 중간), inpainting (왼쪽 아래), super-resolution (오른쪽 아래 2세트) 과제에 pixel2style2pixel을 적용한 결과이다.