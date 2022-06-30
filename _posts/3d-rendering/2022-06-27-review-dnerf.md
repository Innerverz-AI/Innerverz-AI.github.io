---
layout: post
title:  "[Paper Review] D-NeRF: Neural Radiance Fields for Dynamic Scenes"

author: 조우영
categories: [Deep Learning, Multiple-View, Volume Rendering, Neural Radiance Fields, Video]
image: assets/images/logo-3d-rendering.jpeg
---

# Contributions
- **본 논문에서는 single camera를 이용하여 물체의 rigid, non-rigid motion에 대한 정보를 Volume rendering modeling 할 수 있는 방식을 제안한다.** 구체적으로 scene을 canonical space로 encode해주는 모듈과 canonical representation을 deformed scene으로 mapping해줄 수 있는 모듈이 동시에 학습된다. 그 효과로는 camera view, time variable, object movement등 explicit control이 가능해졌다. 실험적으로, video based rigid, non-rigid motion이 담긴 object를 성공적으로 reconstruction할 수 있음을 보였다.

![motivation](/assets/posts/3d-rendering/dnerf/motivation.PNG)

# Related Works

- 기존의 neural scene representation networks들은 static objects에 대해 focus를 맞춰왔다고 논문에서 이야기하고 있다. 하지만, **본 논문은 dynamic scene임을 고려하여 still and moving/deforming object까지 모두 다룰 수 있는 방법을 제안했다.** 또한, 3D ground truth나 multi-view camera setting 혹은 이에 수반하는 calibration approach 없이 monocular data로 구현할 수 있음을 강조한다.

![motivation2](/assets/posts/3d-rendering/dnerf/motivation2.PNG)

- 위와 같이 움직이는 물체에 대하여 각기 다른 time에 대해 다른 view point에서 수집한 data를 활용하여, 새로운 time, viewing direcition에서 본 이미지를 생성해낼 수 있다.
- 저자는 단순히 기존 NeRF input에 time t를 추가하여 실험해보았지만 만족스러운 결과를 얻지 못했다고 서술하고 있다. 이후 classical 3D scene flow에서 영감을 받아 3D spatial mapping and transform mapping을 할 수 있는 2가지 모듈을 제시하였다.

# Dynamic Neural Radiance Fields

![overall-pipeline](/assets/posts/3d-rendering/dnerf/overall-pipeline.PNG)

- 위 그림은  D-NeRF의 overall pipeline이다. **본 pipeline은 a deformation network $\Psi_t$와 a canonical network $\Psi_x$로 구성되어 있다.** $\Psi_t$는 모든 deformed scenes를 canonical space로 mapping해주는 역할을 하고, $\Psi_x$는 입력된 정보를 이용하여 volume rendering을 수행하는 네트워크이다. 두 네트워크가 수행하는 역할에 대한 수식은 아래와 같다.

$$ \Psi_t:(\textbf{x}, t) \to \triangle \textbf{x}, \; \; \Psi_x:(\textbf{x}+\triangle \textbf{x}, \textbf{d}) \to (\textbf{c}, \sigma) $$

- 위 두 모듈이 수행하는 역할에 따라 아래 식과 같이 volume rendering을 진행한다.

$$ C(p,t) = \int_{h_n}^{h_f}\tau (h,t)\sigma(\textbf{p}(h,t))\textbf{c}(\textbf{p}(h,t), \textbf{d})dh $$

기존 NeRF rendering 방식에서 t가 추가된 식이다. 식 이해를 위하여 $\textbf{p}(h,t)$에 주목을 해보자면 아래 식과 같다.

$$ \textbf{p}(h,t)=\textbf{x}(h)+\Psi_t(\textbf{x}(h),t), [\textbf{c}(\textbf{p}(h,t), \; \textbf{d}), \sigma(\textbf{p}(h,t))] = \Psi_x(\textbf{p}(h,t),\textbf{d}) $$

Deformation network로 인해 예측된 transformation 정보 $\triangle \textbf{x}$가 위치정보에 더해지고, 변형된 위치정보를 기반으로 canonical scene에 대한 color와 density를 출력한다.

$$ \tau(h,t) = exp(-\int_{h_n}^{h}\sigma(\textbf{p}(s,t))ds) $$

위 식은 transmittance를 나타내는 식이다. Discrete한 환경의 volume rendering을 위해 변형해주는 방법은 기존 NeRF의 방법과 같다.

이하 volume rendering에 대한 loss 수식은 아래와 같다.

$$ L = \frac{1}{N_s}\sum_{i=1}^{N_s}\left\| \hat{C}(p,t)-C'(p,t)\right\|  $$

$\hat{c}$는 pixel의 ground truth color값을 의미한다.

# Experiments

## Datasets

- Table 1에서 확인할 수 있듯이 8개의 dynamic scene에 대해 training, test를 해보았다.

#### Quantitative results

![result-table1](/assets/posts/3d-rendering/dnerf/result-table1.PNG)

위 table은 vanila nerf, vanila nerf의 input에 단순히 t를 concat하여 rendering한 경우 (T-NeRF), ours의 결과를 정성적인 지표로 비교해본 표이다. 8가지 dynamic scene에 대하여 MSE loss를 포함한 image quality에 대한 지표를 활용하여 평가해보니, D-NeRF의 성능이 가장 좋았음을 말하고 있다.

#### Qualitative results

![result-fig1](/assets/posts/3d-rendering/dnerf/result-fig1.PNG)

가로 열 기준으로 canonical scene, deformed scene (t=0.5, t=1.0)이다. 각각 rgb와 volume density를 이용하여 mesh, depth map, color-coded points를 mapping해본 결과이다. color-coded point는 각 corresponding point가 사물의 움직임에 따라 일관되게 잘 움직이는지를 확인하기 위함이다. 따라서 각 해당 부분의 color가 움직였을 때의 color가 잘 matching이 되는지를 확인하면 된다.

![result-fig2](/assets/posts/3d-rendering/dnerf/result-fig2.PNG)

위와 비슷하게 각 해당하는 point가 time t가 흐름에 따라 잘 matching이 되는지를 확인하기 위함이다. 그림에서도 볼 수 있듯이 잘 matching이 되는 것을 확인할 수 있다.

![result-fig3](/assets/posts/3d-rendering/dnerf/result-fig3.PNG)

close-up을 해보았을 때, D-NeRF가 blurry artifact가 덜했고 ground truth와 제일 가까운 이미지를 생성해냈다.

![result-fig4](/assets/posts/3d-rendering/dnerf/result-fig4.PNG)

![result-fig4-1](/assets/posts/3d-rendering/dnerf/result-fig4-1.PNG)

여러 scene에 대해 Time & View conditioning을 해본 결과이다. canonical space를 기준으로 object의 움직임 뿐만아니라 viewing direction을 다르게 부여할 수 있었다.

# Conclusion

- 기존 NeRF static object assumption에서 벗어나 rigid, non-rigid motion에 대해 novel view synthesis를 할 수 있는 방법을 제안했다. 단순히 time $t$ 정보를 NeRF input에 concat한 것이 아니라, 네트워크를 2개의 모듈로 나누어서 전략적으로 학습할 수 있는 방법을 제안하였다.

# Discussion

- Video based research를 하던 사람이 NeRF를 응용하여 연구한 느낌이 매우 강하게 다가왔다. Video 혹은 multi-frame processing의 관련 분야에서도 항상 baseframe을 기준으로 다른 frame과 align을 하거나 이미지 처리를 하는 경우가 많았는데, 이러한 flow가 반영되었던 것 같다.

- 다만, 아쉬웠던 점은 논문에서 Dataset에 대한 구성은 다소 설명이 부족했던 것 같다. 이는 real-world image에서 실제로 데이터를 구성하기 어려웠을 것이라는 예상이 들었다. 실제로 논문에서 보여준 그림들도 synthetic data로 evalution한 결과로 보였다. 하지만 이후, Nerfie라는 논문에서 realistic deformable scene에 대해 volume rendering한 연구가 있다.  
