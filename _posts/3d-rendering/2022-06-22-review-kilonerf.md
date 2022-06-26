---
layout: post
title:  "[Paper Review] KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs"

author: 조우영
categories: [Deep Learning, Multiple-View, Volume Rendering, Neural Radiance Fields, Fast Inference]
image: assets/images/logo-3d-rendering.jpeg
---

# Contributions
- 새로운 view point에서 관찰한 이미지를 합성시키는 task (Novel View Synthesis: NVS)에서 NeRF는 각 이미지의 픽셀 단위에서 ray를 정의하며, ray 선 상에 존재하는 N개의 point를 hierarchical sampling 후 rendering한다. 하지만 이러한 과정에서 rendering time이 많이 소요되는 것은 NeRF의 고질적인 문제점으로 제기되어 왔다. KiloNeRF는 하나의 MLP를 활용하는 것이 아니라, **여러 소형의 MLP를 다수 구성하여 Knowledge Distillation 방법으로 rendering을 해본 결과: rendering time을 speed up 할 수 있음과 동시에 기존 NeRF와 comparable한 성능을 낼 수 있음**을 실험을 통해 보였다.

![motivation](/assets/posts/3d-rendering/kilonerf/motivation.PNG)

# Related Works

- Faster NeRF rendering을 하기위한 기존 여러 방법들이 존재했다. 대표적으로 Neural Sparse Voxel Fields (NSVF)에서는 3D bounded scene을 uniform 3D voxel grid로 표현하여 rendering time을 줄일 수 있었다. 하지만 NSVF에서는 a single feature conditioned network를 활용하였기 때문에, entire scene을 표현하기 위한 network capacity를 줄이기는 어려웠다. KiloNeRF는 thousands of small networks를 이용하여, 각 network가 small region을 담당하도록 만들어 network의 lower capacity를 구현할 수 있었다.

# KiloNeRF

## Volumetric representation in 3D grid cell (AABB)

- 본 논문에서는 3D object scene의 구역을 나누기 위해 an axis aligned bounding box (AABB) 방법을 이용한다.

![aabb](/assets/posts/3d-rendering/kilonerf/aabb.PNG)

- 위 그림과 같이 minimum bound, maximum bound가 정의되어 있고 a uniform grid of resolution $\textbf{r}$로 grid를 나누어 준다. $\textbf{b}$는 3d vector 좌표계로 표현되어 있고 vector analysis의 개념에서 spatial binning을 진행한 아래 식을 쉽게 이해할 수 있다.

$$ g(\textbf{x}) = \left \lfloor (\textbf{x} - \textbf{b}_{min})/((\textbf{b}_{max} - \textbf{b}_{min})/\textbf{r}) \right \rfloor $$

- Allocated repective grid cell은 이후 rendering 될 color와 density값을 출력하게 된다.

$$ (\textbf{c}, \sigma) = f_{\theta(g(\textbf{x}))}(\textbf{x}, \textbf{d}) $$

## Network Architecture

![architecture](/assets/posts/3d-rendering/kilonerf/architecture.PNG)

- 위 그림은 original NeRF MLP를 tiny MLP로 바꾼 그림이다. 그림에서 볼 수 있다시피, 기존 MLP와 model configuration이 바뀐 것을 확인할 수 있다. 구현을 위해 더 자세한 configuration을 알고자 한다면 본 논문을 참고하는 것이 좋을 것 같다.

## Training with Distilation

![distilation](/assets/posts/3d-rendering/kilonerf/distillation.PNG)

- KiloNeRF를 scratch부터 training한 결과는 위그림의 (a)와 같이 artifact가 생긴다는 것을 본 저자는 실험을 통해 확인했다고 한다. 또한 ordinary NeRF를 teacher모델로 활용하여 tiny NeRF에 Knowledge Distilation을 해준 (b)의 결과를 보면 더욱 less arifact한 결과를 볼 수 있었다.
- 구체적인 방법으로는 각각의 teacher, student 모델에서 출력되는 $\alpha$-values와 color values ($\textbf{c}$)를 같도록 $L_2$ loss를 걸어줘서 student's model의 parameters를 optimize했다.
- $\alpha$-value의 유래는 기존 NeRF가 출력하는 $\textbf{c}$와 $\sigma$ 값에 의존한다.

$$ \hat{\textbf{c}} = \sum_{i=1}^{K}T_i\alpha_i\textbf{c}_i, \; \; \;  T_i = \prod_{j=1}^{i-1}(1-\alpha_j), \; \; \; \alpha_i = 1-exp(\sigma_i\delta_i) $$

### Regularization

- **Model capacity를 바꿀 때에는 inductive bias를 필수적으로 고려해야한다.** 이를 맞춰주기 위해 저자는 처음에 the last hidden layer의 output feature 수를 줄여보는 식으로 KiloNeRF model을 구성해보았으나, 결과적으로 visual loss로 이어짐을 알 수 있었다. 따라서 본 저자들은 view-dependent modeling of the color 부분을 맡고 있는 the last two layers of the network의 weights와 biases들을 $L_2$ regularization을 걸어주었다. 위 방법을 따른 결과, visual quality에 대한 손실 없이, 기존 NeRF와 같은 inductive bias를 같도록 유도하였다.

## Sampling

- 본 논문에서는 ordinary NeRF가 제안했던 hierarchical sampling을 활용하지 않고, **empty space skipping (ESS)** 와 **early ray termination (ERT)**를 활용하여 rendering time을 더욱 줄일 수 있었다고 한다.

### Empty Space Skipping

- Given AABB grid에서 해당 grid가 contents인지 아닌지를 알려주는 binary values를 할당한다. 이후 rendering을 할 때, contents가 아닌 부분에 대해서는 rendering을 skip하여 rendering time을 더욱 빠르게 가져갈 수 있도록 구현하였다.

### Early Ray Termination

- Ray 선상에서 transmittance value $T_i$가 0이 되었을 때에는 굳이 sampling을 할 이유가 없으므로 $T_i$가 일정 threshold 값보다 작으면 sampling을 중단하는 작업을 지칭한다.

# Experiments

## Datasets

- Bounded scene으로 가정되어 있는 setting에서 실험하였다. (Synthetic-NeRF, Synthetic-NSVF, BlendedMVS and Tanks&Temples) 이후 25개 scene에 대해 평가하였다.

#### Quantitative results

![result-table1](/assets/posts/3d-rendering/kilonerf/result-table1.PNG)

여러 scene에 대해서 ordinary NeRF, NSVF, 본 논문의 NVS 성능을 비교해본 표이다. 이미지의 quality를 나타낼 수 있는 PSNR, SSIM, LPIPS 지표를 사용하여 평가해보았다. 특정 scene에서 기존 baseline model에 비해 떨어지는 성능을 보이기도 했으나, 전반적으로 좋은 성능이 나옴을 강조하고 있다. 또한 main contribution인 speed up하는 평가 지표에 있어서는 압도적으로 우월했다.

![result-table2](/assets/posts/3d-rendering/kilonerf/result-table2.PNG)

Empty Space Skipping와 Early Ray Termination를 ordinary NeRF에 추가해보고 본 full model과 비교해본 표이다. 본 모델의 rendering speed 성능이 매우 좋게 나옴을 확인할 수 있었다.

#### Qualitative results

![result-fig1](/assets/posts/3d-rendering/kilonerf/result-fig1.PNG)

![result-fig2](/assets/posts/3d-rendering/kilonerf/result-fig1-1.PNG)

Baseline: NeRF, NSVF와 정성적으로 비교해본 실험으로 detail적 측면, rendering time 측면에서 본 논문의 모델 성능이 더 좋게 나옴을 말하고 있다.

![result-fig2](/assets/posts/3d-rendering/kilonerf/result-fig2.PNG)

Ablation study를 진행해본 정성적인 결과이다. 왼쪽부터 single tiny MLP만 활용하였을 때, network grid size를 반으로 줄여보았을 때($10 \times 16 \times 10$) => ($5 \times 8 \times 5$), network의 hidden unit을 반으로 줄여보았을 때, no fine tuning with training images, full model에 대한 결과이다.

위 실험을 통해 본 저자가 제안한 model 혹은 grid configuration이 합당하다는 것을 이야기하고 있고, 특히 fine-tuning의 필요성이 매우 높다고 강조하고 있다.

![result-fig3](/assets/posts/3d-rendering/kilonerf/result-fig3.PNG)

본 논문에서 제안한 regularization term의 효과로써 scene reconstruction을 할 때 발생할 수 있는 artifact를 avoid했다고 말한다.

# Conclusion

- 하나의 MLP model을 활용했던 기존 NeRF에 반해, 수천개의 작은 MLP를 구성하고 각 모델이 scene의 specific region을 담당할 수 있도록 만들어주어 rendering time을 대폭 줄일 수 있었다. 또한 Knowledge Distilation 이론을 활용하여 기존 baseline 모델에 대해 comparable한 성능을 낼 수 있음을 실험을 통해 보였다.

# Discussion

- 논문에서도 언급하다시피 3D object scene을 표현하기 위해 3D voxel grid의 형식으로 표현하기 위해 AABB 방법을 사용하였다. 하지만  minimum bound, maximum bound를 활용하였으므로 unbounded scene에 대해서 표현하기는 어려웠다. Novel view synthesis를 하기 위한 dataset에는 unbounded scene에 대한 case도 존재하므로 추후에 unbounded scene 합성과 reducing rendering time에 대한 연구도 진행될 수 있을 것이다.
