---
layout: post
title:  "[Paper Review] Pose-Controllable Talking Face Generation by Implicitly Modularized Audio-Visual Representation (PC-AVS)"
author: 이유경
categories: [PC-AVS, pose controllable Lip-sync, Neural Talking Face]
image: assets/images/logo-face-reenactment.jpeg
---

![main-figure](/assets/posts/face-reenactment/2022-05-23-reivew-pc-avs/main-figure.png)

# Contributions
- 각각 speech content, head pose, identity information에 관한 세 종류의 representation을 사용해서 talking human face를  modularize 할 수 있는 방법을 제안했다. 
- StyleGAN2 generator를 사용한다.
- Head pose를 자유롭게 조절할 수 있는 lip sync 모델이다. 
- Landmark 같은 structural intermediate를 전혀 사용하지 않는다. 즉, pre-processing이 거의 필요하지 않다.

# Our Approach
아래 그림은 전체 framework를 보여준다. 여기에 등장하는 모든 encoder는 ResNet 구조를 기반으로 한다.

![framework](/assets/posts/face-reenactment/2022-05-23-reivew-pc-avs/framework.png)

학습 단계에서 PC-AVS 모델은 $K$-frame video $V = \\{I_{(1)}, \cdots,  I_{(K)}\\}$와 그에 대응하는 spectrogram $A = \\{S_{(1)}, \cdots , S_{(K)}\\}$을 입력받는다. 그리고 video frame 중 하나를 identity reference $I_{(ref)}$로 사용한다. 

## Identifying Non-Identity Feature Space
Non-identity space 위의 점에는 pose나 facial movement에 관한 정보가 담긴다. $k$번째 target frame $I_{(k)}$로부터 texture나 facial structure 등 identity에 관한 정보는 무시하고 싶다. 따라서, 먼저 augmentation을 ${I'}_{(k)}$를 구한다. 아래 그림의 왼쪽에 그 과정이 잘 드러나 있다.

![augmentation](/assets/posts/face-reenactment/2022-05-23-reivew-pc-avs/augmentation.png)

덧붙이자면, perspective transformation이란 이미지를 포함하는 직사각형 $P_s$ 영역을 등변 사다리꼴 $P_t$ 영역으로 변환하는 과정을 의미하고, color transfer란 RGB channel의 순서를 섞는 과정을 의미한다. 이렇게 얻은 augmented target frames $ V^\prime = \\{I^\prime_{(1)}, \cdots,  I^\prime_{(K)} \\} $는 encoder $E_n$을 통과하면서 non-identity space 위의 feature $F_n = \\{ f_{n(1)}, \cdots,  f_{n(K)} \\}$으로 변한다.

(한 video에 있는 frame들이 모두 같은 방법으로 augmentation 되는 건지는 잘 모르겠다 ...)

## Modularization of Representations
### Speech Content Space
<span style="color:red"> 
Tip. 위에 있는 framework 그림에서 파란색으로 표시된 부분을 참고하자.
</span>  

앞서 구한 non-identity feature $F_n$를 MLP에 통과시켜서 visual speech content feature $F_c^v$를 얻는다. 그리고 spectrogram $A = \\{ S_{(1)}, \cdots , S_{(K)} \\}$를 encoder $E_c^a$에 통과시켜서 audio feature $F_c^a$를 얻는다. **저자들은 두 speech content feature, $F_c^v$와 $F_c^a$ 모두 하나의 space 위에 있고, aligned pair일 때 distance가 가깝다고 가정한다.** 

Space에 대해 학습하기 위해서 contrastive learning 전략을 사용한다. 쉽게 설명하자면, 어떤 $F_c^v$에 대해서 positive pair인 $F_c^a$ 외에도 negative pair $F_c^{a-}$를 $N^-$개 준비한다는 뜻이다. Negative audio는 다른 video나 같은 video를 time-shift해서 얻을 수 있다. 따라서 video-to-audio synchronization loss는 아래와 같이 정의된다. $\mathcal{D}$로는 가까울수록 큰 값을 가지는 cosine distance를 선택했다. 

$$ \mathcal{L}_c^{v2a} = - \log [\frac
{\exp(\mathcal{D}(F_c^v, F_c^a))}
{\exp(\mathcal{D}(F_c^v, F_c^a)) + \sum_{j=1}^{N^-} \exp(\mathcal{D}(F_c^v, F_c^{a-}))}]$$

$$ \mathcal{D}(F_1, F_2) = \frac {F_1^T * F_2}{\vert F_1\vert \cdot \vert F_2\vert}$$

비슷한 방법으로 audio-to-video synchronization loss도 계산할 수 있다. 최종적으로는 두 loss의 합을 이용해서 speech content space를 encoding한다. 

$$ \mathcal{L}_c = \mathcal{L}_c^{v2a} + \mathcal{L}_c^{a2v} $$

### Pose Code
<span style="color:red"> 
Tip. 위에 있는 framework 그림에서 노란색으로 표시된 부분을 참고하자.
</span>  

Non-identity feature $F_n$를 MLP에 통과시켜서 12-dim. 3D pose code $F_p$를 얻는다. **Pose code를 이루는 12개 숫자 중에서 9개는 3D rotation matrix, 2개는 2D positional shifting bias, 1개는 scale factor에 대응된다.** 뒤에서 소개할 ablation study에서 pose code의 길이를 36으로 늘려봤는데, 오히려 불필요한 정보들까지 섞이면서 합성 샘플의 품질이 낮았다. 

### Identity Space
<span style="color:red"> 
Tip. 위에 있는 framework 그림에서 빨간색으로 표시된 부분을 참고하자.
</span>  

Target video $V$를 encoder $E_i$에 통과시켜서 identity feature $F_i = \\{ f_{i(1)}, \cdots,  f_{i(K)} \\}$를 얻는다. 학습에 사용한 Voxceleb2 dataset에는 GT identity label이 이미 있다. 따라서 $F_i$와 GT label에 대해서 softmax cross-entropy loss $\mathcal{L}_i$를 적용한다. 

(Framework 그림을 보면 $f_{i(ref)}$를 $K$번 반복해서 사용하는 것 같기도 하고 ...)

# Talking Face Generation
<span style="color:red"> 
Tip. 위에 있는 framework 그림에서 보라색으로 표시된 부분을 참고하자.
</span>  

**StyleGAN2 generator를 사용한다!** Wav2Lip의 unet-like generator처럼 skip connection이 있는 generator를 쓰면 pose 변화에 제약이 생긴다. 위에서 소개한 세 가지 representation을 concatenate 한 다음 MLP에 통과시켜서 latent code를 얻을 수 있다.

## Training
여느 face reenactment 모델에서처럼 PV-AVS 또한 video reconstruction task를 가지고 학습한다. 먼저 multi-scale discriminator를 사용해서 $N_D$개 layer로부터 feature map L1 loss를 계산했고 최종 출력으로 adversarial loss를 계산한다. 

$$\mathcal{L}_{L_1} = \sum_{n=1}^{N_D} \| D_n(I_{(k)}) - D_n(G(f_{cat(k)})) \|_1$$

$$\mathcal{L}_{GAN} = \min _G \max _D \sum_{n=1}^{N_D} ( 
\mathbb{E}_{I_{(k)}} [\log D_n(I_{(k)})] 
+ \mathbb{E}_{f_{cat(k)}} [\log (1 - D_n(G(f_{cat(k)})))] )$$

그리고 pretrained VGGNet의 $N_P$개 layer에서 얻은 feature map으로 perceptual loss를 계산한다. 

$$\mathcal{L}_{vgg} = \sum_{n=1}^{N_P} \| {VGG}_n(I_{(k)}) - {VGG}_n(G(f_{cat(k)})) \|_1$$

따라서 total objective는 아래와 같이 나타난다. 이전 section에서 소개한 space 학습에 관련된 loss도 찾아볼 수 있다. 

$$\mathcal{L}_{total} = \mathcal{L}_{GAN} + \lambda_1 \mathcal{L}_{L_1} + \lambda_v \mathcal{L}_{vgg} + \lambda_c \mathcal{L}_c + \lambda_i \mathcal{L}_i$$

# Experiments
아래 그림은 다른 모델들과 성능을 비교한 결과를 보여준다. 

## Model Comparison

![model-comparison](/assets/posts/face-reenactment/2022-05-23-reivew-pc-avs/model-comparison.png)

- Wav2Lip은 lip sync 정확도는 높지만 한 장의 $I_{(ref)}$만 입력받아서는 static한 샘플 밖에 합성할 수 없다는 한계가 있다. 
- MakeitTalk 이후 모델들은 head motion을 합성하는 것이 가능하다. 그러나 MakeitTalk는 lip sync 정확도가 낮고, Rhythmic Head는 identity를 잘 유지하지 못했다. 
- 따라서 PV-AVS로 합성한 샘플이 SoTA 모델들과 비교했을 때 더 품질이 좋았다.

## Ablation Study
Ablation study에서는 low-dim. pose code가 더 강력하다는 점에 주목하자. 

![ablation-study](/assets/posts/face-reenactment/2022-05-23-reivew-pc-avs/ablation-study.png)

## Under Extreme Condition
Identity reference $I_{(ref)}$에서는 보이지 않는 각도를 포함하는 target video $V$를 넣어줬다. 아래 그림은 $I_{(ref)}$로 얼굴의 왼쪽 절반을 보여준 다음, 정면을 보는 $V$를 준 경우의 결과이다. 모든 frame에 대해 zero vector를 pose code로 사용하면 완전 정면을 보는 샘플까지도 합성가능했다.

![extreme-condition](/assets/posts/face-reenactment/2022-05-23-reivew-pc-avs/extreme-condition.png)

# Conclusions
- Modularization 전략은 성공적이었다. 
- Talking face의 pose를 자유롭게 조절할 수 있다.
- Extreme condition에 대해서도 강력하다. (아마 warp-based가 아니고 StyleGAN2 generator를 사용해서 그런게 아닐까?)
