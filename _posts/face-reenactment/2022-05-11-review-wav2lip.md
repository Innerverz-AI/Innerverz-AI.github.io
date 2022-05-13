---
layout: post
title:  "[Paper Review] A Lip Sync Expert Is All You Need for Speech to Lip Generation In The Wild (Wav2Lip)"
author: 이유경
categories: [Wav2Lip, LipGAN, SyncNet, Lip-sync, Neural Talking Face]
image: assets/images/logo-face-reenactment.jpeg
---

Wav2Lip의 선행 모델인 [LipGAN을 소개한 글](https://yukyeongleee.github.io/2022-05-10/review-lipgan/)을 먼저 읽고 오는 것을 추천한다. 

# Contributions
- 임의의 얼굴(speaker id)과 음성(speech)으로 합성할 수 있다.
- speaker-independent 모델 중에서 처음으로 real synced video와 비슷한 정확도로 영상을 합성한다. 합성된 영상에서 out-of-sync segment가 차지하는 비율이 10% 이내이다. 
- Lip sync 과제에 대한 새로운 evaluation framework를 제안했다. 기존에 사용된 방법들과는 다르게 wrong lip에 대해 적절한 penalty를 부여했고, 강력한 lip sync discriminator를 도입했다. 

# Lip Sync in the Wild
## Related Works
논문이 나올 당시에는 얼굴, 음성, 언어에 제약을 받지 않는 lip sync 모델이 거의 없었다. 그나마 Speech2Vid(BMVC 2017)과 LipGAN(ACMMM 2019)이 있었는데, 두 모델 모두 L1 reconstruction loss를 사용했고 추가적으로 LipGAN은 adversarial loss를 사용했다. 저자들은 기존 방법들로 **wrong lip에 대해 적절한 penalty를 부여할 수 없다**고 주장했다. 그 이유를 요약하자면 이렇다.
- 얼굴 이미지에서 입술이 차지하는 비율은 약 4%이다. 따라서 reconstruction loss 만 사용했을 때, 전체 학습 시간의 절반이 지나서야 입술 영역이 morphing 되기 시작한다.  
- LipGAN의 discriminator는 out-of-sync를 잡아내는 정확도가 56%로 꽤 낮다. 아래 두 가지가 원인이다.
  - Discriminator는 generated frame을 하나씩 입력받기 때문에 temporal context를 잘 활용하지 못한다. 
  - GAN setup으로 generator와 discriminator를 동시에 학습시키면, 초반에 만들어지는 noisy frame 때문에 discriminator가 audio-lip correspondence보다 visual artifact에 집중하게 될 가능성이 높다. 

## Lip Sync Expert 
LipGAN의 실패에서 교훈을 얻어 **pretrained expert lip sync discriminator**를 도입했다. 즉, generator와 discriminator를 동시에 학습시키지 않는다. 구조는 lip sync error를 계산하도록 설계된 SyncNet(ACCV 2016)을 변형해서 가져왔다. 

### SyncNet (ACCV 2016)
![syncnet-architecture](/assets/posts/face-reenactment/2022-05-11-review-wav2lip/syncnet-architecture.png)

SyncNet은 $T_v(=5)$개의 grayscale face frame과 $T_a(=20)$개의 audio frame을 입력받아서 각각을 encoding한다. 그런 다음 두 embedding 사이의 L2 distance를 계산해서 출력한다. 모델을 학습시킬 때는 아래 식처럼 정의된 max-margin loss를 사용했다. 어제 리뷰했던 LipGAN이 SyncNet을 많이 참고한 것 같다.

$$ E = \frac {1}{2N} \sum_{n=1}^{N}{[y_n d_n^2 + (1 - y_n) \max(margin - d_n, 0)^2]} $$

$$ d_n = \Vert v_n - a_n \Vert _2 $$

### Our Expert Lip Sync Discriminator
SyncNet에서 수정한 부분은 이렇다. 
- grayscale 대신에 3-channel 이미지를 사용한다.
- Conv2D를 더 많이 쌓는 대신에 residual skip connection을 추가한다.
- Loss function을 아래와 같이 정의된 cosine-similarity with BCE로 대체한다. 

$$ P_{sync} = \frac {v \cdot s}{\max(\Vert v \Vert_2 \cdot \Vert s \Vert_2, \epsilon)}$$

$$ E_{sync} = \frac {1}{N} \sum_{i=1}{N} -log(P_{sync}^i) $$

이렇게 학습한 discriminator는 LRS2 dataset에 대해 91%의 정확도로 out-of-sync를 잡아냈다. 

# Architecture
Wav2Lip은 generator와 두 개의 discriminator(expert lip sync, visual quality)로 이루어져있다. Visual quality discriminator는 GAN setup 아래에서 generator와 동시에 학습시킨다.
![architecture](/assets/posts/face-reenactment/2022-05-11-review-wav2lip/architecture.png)

- LipGAN의 후속 모델이니만큼 generator를 거의 그대로 가져와서 사용한다. 
- Visual quality discriminator는 Conv2D가 여러 개 쌓인 단순한 구조를 하고 있다. 앞에서 GAN setup으로 학습시켜서는 audio-lip correspondence가 잘 고려되지 않는다고 언급했는데, 이 discriminator의 역할은 generated frame이 photo-realistic하도록 유도하는 것이라서 상관없다. 

# Objectives
학습에 사용한 objective 역시 LipGAN과 거의 비슷하다. 
## Adversarial Loss
아래 정의에서 $D$는 visual quality discriminator를 가리킨다. 그리고 $L_g$는 generated frame, $L_G$는 real image를 의미한다. 

$$ L_{gen} = \mathbb{E}_{x \sim L_g} [log(1-D(x))] $$

$$ L_{disc} = \mathbb{E}_{x \sim L_G} [log(D(x))] + L_{gen} $$

## L1 Reconstruction Loss
마찬가지로 generator 학습을 돕기 위해서 **L1 reconstruction loss**를 함께 사용한다. 

$$ L_{recon}(G) = \frac{1}{N} \sum_{i=1}^{N} \Vert L_g - L_G \Vert_1 $$

## Total Loss
Generator의 최종 objective는 다음과 같다. Pretrained expert discriminator로 평가한 error $E_{sync}$도 함께 추가되었다.

$$ L_{total} = (1 - s_w - s_g) \cdot L_{recon} + s_w \cdot E_{sync} + s_g \cdot L_{gen} $$

Visual quality discriminator는 조금 전에 설명한 $L_{disc}$를 사용해서 학습시킨다.

# Experiments
![result](/assets/posts/face-reenactment/2022-05-11-review-wav2lip/result.png)
각 모델을 사용해서 합성한 결과가 서로 다른 색으로 표시되어 있다.
- LipGAN → 빨간색 
- Visual quality discriminator를 사용하지 않은 Wav2Lip → 노란색
- Full Wav2Lip → 초록색

확실히 LipGAN보다 Wav2Lip을 사용했을 때 입 모양과 음성 간에 관련도가 높아보인다. 게다가 visual quality discriminator를 도입한 경우에 구강 내부, 치아까지도 더 자연스럽게 표현된다. 

논문에는 resolution에 대한 언급이 없는 것 같아서 official implementaion을 살펴봤다. Wav2Lip으로 전달하기 전에 face alignment network(FAN)라는 오픈 소스 모델을 사용해서 얼굴 영역을 인식(face detection)하고 그 부분을 **96 x 96 x 3**으로 resize하는 것을 확인했다.

# Conclusion
- 선행 모델들의 penalty 부여 방법이 왜 부적절한가에 대한 명쾌한 이유를 제시하고, 이를 해결할 수 있는 확실한 방법을 제시한 점이 좋았다. 