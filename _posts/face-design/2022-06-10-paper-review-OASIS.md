---
layout: post
title:  "[Paper Review] OASIS : You only need adversarial supervision for semantic image synthesis"
author: 정정영
categories: [Deep Learning, Spatially-Adaptive Normalization, OASIS, face-design]
image: 
---

![Author](/assets/posts/face-design/OASIS/1.author.png)


# 1. Contributions
- Discriminator를 Auto Encoder 구조로 바꿔 semantic image가 나오도록 하였다.
- Generator input이 3D Noise와 label image로 구성되어있다.

3D noise를 이용하여 일부 영역만 다르게 변환할 수 있으며, Discriminator 구조를 바꿨기 때문에 Label map region에 충실한 fake image를 생성할 수 있었다.

# 2. Motivation
Semantic image를 이용하여 realistic image를 생성하는 방법은 SPADE에서 한적이 있다. 그러나 SPADE는 semantic label의 경계 부분에서 생성 퀄리티가 낮다는 단점을 가지고 있다. 

# 3. Method
![OASIS](/assets/posts/face-design/OASIS/2.OASIS.png){: width="100%", height="100%"}<br>
## 3.1 The SPADE baseline
SPADE에서 sematic label map을 input으로 받는 spatially-adaptive normalization layer를 synthesis process 곳곳 적용하였다. 그렇게 원하는 영역에 label에 해당하는 물체를 생성할 수 있었다. 또한 SPADE에 스타일을 추출하는 encoder를 추가하면 원하는 스타일을 semantic label map의 region에 적용이 가능하였다.(multi-modal synthesis)    

본 논문은 SPADE architecture를 수정하고 새로운 discriminator를 소개하여 SPADE보다 나은 성능을 보였다.  


## 3.2 OASIS discriminator
SPADE에서 discriminator는 PatchGAN의 multi-scale discriminator를 사용하였다. SPADE discriminator는 입력으로 rgb image와 semantic label map을 입력으로 받고 단순하게 real/fake를 구분하였다. 이는 결과적으로 semantic label map의 정보를 무시하거나 realism이 부족한 결과를 가져왔다.  
 그래서 OASIS는 encoder-decoder segmentation network를 만들었다. 이 network는 U-Net architecture와 비슷하며 input으로 rgb image를 받고 결과로 semantic image를 생성한다. Encoder-decoder 형식의 network로 생성된 semantic image와 semantic label map을 비교하여 SPADE discriminator의 문제점을 해결하고자 하였다.

### Formuler
Semantic label map을 ground true로 정한다. 이 semantic label map은 N개의 label을 가지고 있다고 가정한다. fake image를 OASIS discriminator의 input으로 넣어 N + 1개 class로 분류된 semantic map을 얻는다. (1개 class가 추가된 이유는 class에 없는 index pixel은 extra class로 지정하였기 때문이다.)  

이미지는 보통 N class가 골고루 존재하지 않는다. 이런 점을 감안하여 각 semantic classes 마다 per-pixel size에 따라 다른 weight를 곱하도록 하였다.(드물게 나타나는 class에 대해 높은 weight를 부여하여 학습하였다.) 이렇게 모든 class에 대해 balanced를 유지하게 하였고, generator가 적게 나타나는 class에 대해서도 잘 생성할 수 있도록 하였다.  

#### Discriminator loss
$
L_{D} = -\mathbb{E}_{(x,t)} \left [ \sum^{N}_{c=1} \alpha_{c} \sum^{H*W}_{i,j} t_{i,j,c}logD(x)_{i,j,c} \right ] - \mathbb{E}_{(z,t)}\left [ \sum^{H*W}_{i,j} logD(G(z,t))_{i,j,c=N+1} \right]
$
> - x : real image  
> - (z,t) : noise-label map pair  
> - t : ground truth label map(three dimension)  

추가로 weight c는 다음과 같이 계산된다.
$
\alpha_{c} = \frac{H x W}{\sum^{H*W}_{i,j} E_{t}[\mathbb{I}[t_{i,j,c} = 1]]}
$

### LabelMix regularization
real image와 fake image간 content와 structure 차이를 discriminator가 잘 구분하도록 하기 위해서 LabelMix regularization을 제안한다.  

fake image의 sematic image와 real image의 semantic image를 섞은 semantic image와 fake image와 real image를 섞은 이미지를 semantic image로 만들었을 때 결과가 동일해야 한다는 점을 이용한 방법이다.(동일한 영역을 섞는다고 가정한다.)  

$$
LabelMix(x,\hat{x},M) = M \bigodot x + (1 - M)\bigodot \hat{x}
$$

![LabelMix](/assets/posts/face-design/OASIS/3.LabelMix.png){: width="100%", height="100%"}<br>
> - $LableMix_{(x,\hat{x})}$ 는 Mask M에 따라 1영역은 real image x, 0영역은 fake image을 따라 생성된다.  
> - 만들어진 $LableMix_{(x,\hat{x})}$ 를 discriminator에 넣어줘서 semantic map을 얻는다.  
> - discriminator를 통해 얻은 semantic map($D_{LableMix_{(x,\hat{x})}}$)과 $LabelMix_{(D_{x},D_{\hat{x}})}$ 간 L2 loss를 계산하여 최소화 하도록 discriminator를 학습하였다.  

style을 섞을 이미지가 주어지면 LabelMix operation을 잘 수행하도록 discriminator를 학습시키기 위해서 밑의 $\textit{L}_{cons}$ 를 추가하였다.
$$
\textit{L}_{cons} = \left\| D_{logits}(LabelMix(x,\hat{x},M)) - LabelMix(D_{logits}(x),D_{logits}(\hat{x}),M)) \right\|^{2}
$$

LabelMix regularization을 적용하면 generator가 이미지를 생성할 때 pixel-level로 사실적이게 생성하도록 하며, 자연스러운 semantic boundaries를 얻는 방향으로 학습하게 된다고 한다.

## 3.3 OASIS generator
OASIS discriminator design을 적용한 새롭게 바뀐 generator의 loss를 다음과 같다.
$
L_{D} = -\mathbb{E}_{(z,t)} \left [ \sum^{N}_{c=1} \alpha_{c} \sum^{H*W}_{i,j} t_{i,j,c}logD(G(z,t))_{i,j,c} \right ]
$
> - z : noise tensor(64 * H * W)
> - t : label map(H*W)

generator에 z와 t를 channel-wise concatenation한 tensor를 input으로 넣어준다.(65 * H * W) 이 concatenation한 tensor는 generator의 각 spatially-adaptive normalization layer에도 넣어준다. 이러한 구조로 인하여 generator는 noise-dependent image를 생성하게 된다. 

generator가 noise-dependent image를 생성하기 때문에 해당 class영역의 noise값을 다른 값으로 바꾸면 원하는 영역만 다른 스타일로 변경된 사진을 얻을 수 있다. 

OASIS generator와 SPADE genrator간 다른 점을 소개하자면 네트워크 복잡도를 줄이기 위해서 OASIS는 SPADE generator의 첫번째 residual block를 제거하였다. 이로써 96M개 parameters를 72M개 까지 줄이게 되었다.


# 4. Experiments
- datasets : ADE20K, COCO-stuff, Cityscapes
- pre-trained semantic segmentation network : Uper-Net101 for ADE20K, multi-scale DRN-D-105 for Cityscapes, DeepLabV2 for COCO-Stuff

특이한 점은 OASIS를 학습할 때 feature matching loss를 사용하지 않았다. ablation  할때만 VGG loss를 추가하였다.

![qualitative_comparsion](/assets/posts/face-design/OASIS/4.qualitative_comparsion.png){: width="100%", height="100%"}<br>

![compare_table](/assets/posts/face-design/OASIS/5.compare_table1.png)
- OASIS는 adversarial supervision 만 사용하였는데 좋은 performance를 보인다.
- SPADE+는 VGG loss가 없으니 퀄리티가 갑자기 낮아진다.

### Multi-modal image synthesis
![multi-modal_synthesis](/assets/posts/face-design/OASIS/6.multi-modal_synthesis.png){: width="100%", height="100%"}<br>

# 5. Conclusion
- OASIS를 이용하여 좋은 퀄리티로 image synthesis를 수행할 수 있다.
- 새로운 discriminator를 적용한 덕분에 perceptual loss를 적용하지 않고도 좋은 performance를 보였다.
- 이러한 discriminator를 바탕으로 generator에서 3D noise re-sampling을 통하여 multi-modal output을 얻을 수 있었다.

# 6. Think
저자들은 새로운 방식의 Segmentation-Based Discriminator를 중심으로 논문을 소개하였다. 강력한 feedback을 generator에 줄 수 있는 encoder-decoder 형식의 discriminator를 기반으로 이미지를 잘 생성할 수 있게 되었고 input은 noise와 label map을 채널 단위로 concatenation하였기 때문에 multi-modal synthesis를 달성할 수 있었다. 

다만,
- 원하는 스타일을 부여할 수 없다는 한계점
- 사람, 차와 같이 물건에 존재하는 패턴을 적절하게 표현하지 못한다는 한계점(사람을 예시로 하면, 하체는 바지를 상체는 티를 입고 있는 모습의 패턴을 의미한다.)
이 보인다.