---
layout: post
title:  "[Paper Review] Dynamic Neural Radiance Fields for Monocular 4D Facial Avatar Reconstruction"

author: 조우영
categories: [Deep Learning, Multiple-View, Volume Rendering, Neural Radiance Fields, Video, Facial expressions]
image: assets/images/logo-3d-rendering.jpeg
---

# Contributions
- **본 논문에서는 사람의 얼굴 표정에는 많은 dynamics 혹은 expressions에 대한 정보를 수반할 수 있는 Volume rendering 방식을 제안한다.** 구체적으로 low-dimensional morphable model 이라는 외부 모듈을 이용하여 본 scene representation task에 포함시켰다. 그 효과로는 사람의 얼굴에 있어 pose나 expressions에 대한 explicit control이 가능해졌다. 또한, monocular input data만을 활용하여 본 task를 수행할 수 있음을 저자는 강조하고 있다. 실험적으로, 각종 video based reenactment methods와 성능 비교를 하였고, 본 모델이 그들을 뛰어넘는 성능을 얻을 수 있었음을 보였다.

![motivation](/assets/posts/3d-rendering/nerface/motivation.PNG)

# Related Works

- 기존의 neural scene representation networks들은 static objects에 대해 focus를 맞춰왔다고 논문에서 이야기하고 있다. 하지만, 본 논문은 facial pose, viewing direction 뿐만 아니라 complex facial deformation까지 모두 다룰 수 있는 방법을 제안했다. (필자의 의견은 2021년이 되고 나서야 dynamic scene에 대한 Nerual representation에 대한 문제 제기를 했던 논문들이 쏟아져 나왔기에, 본 논문에서는 다른 dynamic neural scene representation 방법을 언급하지 않은 것으로 보인다.)

# Dynamic Neural Radiance Fields

![overall-pipeline](/assets/posts/3d-rendering/nerface/overall-pipeline.PNG)

- 위 그림은 본 모델인 Nerface의 overall pipeline이다. 먼저 본 네트워크를 학습하기 위해서는 input frame, background without human, 3D morphable model, 각 frame에 해하는 per-frame learnable code $\gamma \in \mathbb{R}^{M\times 32} $가 필요하다. 3D morphable model을 통해 사진 속 human의 pose 정보 $\textbf{p} \in \mathbb{R}^{4\times 4}$, camera intrinsics $(f_{x,y}, c_{x,y}) \in \mathbb{R}^{3\times 3}$, expression $\delta \in \mathbb{R}^{76}$ 를 추출한다. $\textbf{p}$ 와 $(f_{x,y}, c_{x,y})$ 를 통해 viewing ray sampling 작업을 거쳐서 각 pixel에 해당하는 ray를 정의하고, 그 ray의 direction vector $\vec{v}$를 구해 NeRF 모델에 입력한다. 추가적으로 정의해둔 perframe-learnable code $\gamma$ 와 expression $\delta$ 정보를 함께 NeRF 모델에 입력하고, 출력값으로 얻은 color와 density를 기반으로 rendering을 진행한다. 아래 식이 overall pipeline을 간단한 식으로 나타낸 수식이다.

$$ D_\theta(\textbf{p}, \vec{v}, \delta, \gamma) = (RGB, \sigma) $$

- 아래 사진은 추가적으로 facial expression에 관한 semantic control에 관한 실험을 진행해본 것이다. 실제 mouth opening의 blendshape coefficient를 (left -0.4, right +0.4) 조정해보았더니, 사람이 입을 벌리고 다무는 이미지가 합성되었다는 것을 확인 했다. 또한 geometric check를 위해 depth 정보를 이용하여 normal map을 그려본 것이 두번째 행의 결과 이미지 인데, 실제로 mouth의 geometric 구조가 잘 반영되었다고 저자는 주장하고 있다.

![editing](/assets/posts/3d-rendering/nerface/editing.PNG)

## Dynamics Conditioning

- 본 논문에서 사용한 **per-frame learnable code $\gamma$는 canonical image로 transform 할 때, 얼굴 표정의 missing information을 보완하기 위해 제안**되었다. 그 contribution에 대한 실험적 증거로 Fig. 4의 결과와 함께 LPIPS 성능 개선을 보임을 증명했다.

## Volumetric Rendering of Portrait Videos

- 본 논문에서 reenactment task를 구현하는 것에 있어 부가적인 모듈들을 소개하는 문단이다.

- 첫번째로, 3D morphable model으로 부터 얻을 수 있는 transformation matrx $P$를 활용하여 저자들은 test time때 head pose를 control할 수 있도록 만들었다.

- 두번째로, static background와 dynamic object를 분리할 수 있도록 static background에 해당하는 each end of the ray value를 background image value값과 같도록 fix 시켜주었다. 또한 background의 density는 foreground의 density보다 낮기 때문에 이를 쉽게 분류할 수 있고, 이 특성을 이용하여 decouple 시켰다. 이는 head 주변의 background가 blur해지는 현상을 방지할 수 있었다고 저자는 설명한다.

![background](/assets/posts/3d-rendering/nerface/background.PNG)

## Network Architecture and Training

- 본 논문에서 활용한 모델과 training scheme은 기존 NeRF 방법과 매우 유사하다. 따라서 식으로만 정리하고 넘어가고자 한다.

$$ C(\textbf{r};, \theta, P, \delta, \gamma) = \int_{z_{near}}^{z_{far}}\sigma_\theta(\textbf{r}(t))\textbf{RGB}_\theta(\textbf{r}(t), \vec{\textbf{d}})\cdot T(t)dt, \; \;  T(t) = exp(-\int_{z_{near}}^{t}\sigma_\theta(\textbf{r}(s))ds) $$

$$ L_{total} = \sum_{i=1}^{M}L_i(\theta_{coarse}) + L_i(\theta_{fine}) \; \; with \sum_{j \in pixels}^{}\left\| C(\textbf{r}_j;\theta, P_i, \delta_i, \gamma_i)-I_i[j]\right\|^2 $$

# Experiments

## Datasets

- Nikon D5300 으로 수집한 short monocular RGB video sequences로 해상도는 1920 $\times$ 1080 으로 50 fps로 촬영되었다.

- 데이터 전처리로는 1080 $\times$ 1080 으로 crop한 후, 512 $\times$ 512으로 rescale하여 사용한다.

- 데이터에는 사람이 normal conversation 하는 장면이 담겨 있고 smiling expression 또한 포함되어 있다.


#### Quantitative results

![result-table1](/assets/posts/3d-rendering/nerface/result-table1.PNG)

위 table은 최신 facial reenactment 모델들과 정성적인 지표로 비교해본 결과이다. no BG는 background에 대한 condition을 고려하지 않았을 때의 결과이고, no dyn.은 learnable parameter를 사용하지 않았을 때의 결과이다. Nerface의 full model의 성능이 가장 좋게 나오는 것을 확인할 수 있다.

![result-table2](/assets/posts/3d-rendering/nerface/result-table2.PNG)

Training dataset size를 바꿔가며 실험한 것이다. 데이터 셋을 각각 25%, 50%, 100% 활용해보았을 때의 결과이고, 다양한 case를 담은 large corpus training dataset을 활용했을 때 결과가 가장 좋다.

#### Qualitative results

![result-fig1](/assets/posts/3d-rendering/nerface/result-fig1.PNG)

![result-fig1-1](/assets/posts/3d-rendering/nerface/result-fig1-1.PNG)

Facial reenactment를 진행해본 결과 DNR 모델은 head pose를 control할 수 없었고, FOMM, DVP는 faithful result를 얻기 어려웠다. 반면 Nerface는 facial expression과 appearance especially eye glasses 부분이 잘 생성되었음을 실험을 통해 알 수 있었다.

![result-fig2](/assets/posts/3d-rendering/nerface/result-fig2.PNG)

Ground truth와 다른 새로운 head pose나 expression 정보도 3d morphable model을 활용하였기에 user-defined control을 할 수 있었다.

![result-fig3](/assets/posts/3d-rendering/nerface/result-fig3.PNG)

Head pose와 facial expression을 원하는 avatar id에 transfer도 할 수 있었다. 구제적인 방법은 논문에 나와있지는 않지만, 필자의 생각으로는 reference(facial expression & pose를 담고 있는 이미지)로 부터 3d morphable model을 활용하여 pose, expression 정보를 추출한 후, NeRF model이 target actor를 rendering할 수 있게끔 만들어준 것으로 보인다.

# Conclusion

- 3d morphable model을 활용하여 facial pose, expression에 대한 정보를 control할 수 있도록 만들어주었고, 추가적으로 learnable latent code, decoupling foreground and background의 technique을 활용하여 realistic results를 얻을 수 있는 것에 기여하였다.

# Discussion

- 논문에서도 언급하다시피 eye blinking에 관한 issue는 아직 해결하지 못했다고 한다. Eye blinking도 facial expression의 요소 중 하나이므로 이런 case까지 고려한다면 더욱 좋아질 것으로 보인다. 또한 video frame에 pair가 되는 background 이미지가 있어야 하는데, 이러한 제한적 요건으로 범용적인 활용성에 있어서는 다소 떨어져 보인다. 그러나 object가 가리고 있는 background를 inpainting 방법을 통해 채운다거나 다양한 방법으로 그 제한성을 메꿀 수 있을 것 같다.
