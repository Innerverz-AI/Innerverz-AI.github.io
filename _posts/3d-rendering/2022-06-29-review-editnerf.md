---
layout: post
title:  "[Paper Review] Editing Conditional Radiance Fields"

author: 조우영
categories: [Deep Learning, Multiple-View, Volume Rendering, Neural Radiance Fields, Editing]
image: assets/images/logo-3d-rendering.jpeg
---

# Contributions
- **본 논문에서는 category-level로 user-defined object editing을 할 수 있는 Volume rendering modeling 방식을 제안한다.** 구체적으로 object의 color와 shape을 editing하기 위해 각 part를 해당하는 color branch, shape branch를 나누었고, 두 모듈이 conditionally 학습된다. 그 효과로는 color와 shape에 대한 editing이 가능해졌을 뿐만아니라, 모델이 same categorie내의 수많은 instance를 보며 어떠한 semantic supervision 없이 category-level editing이 가능했다라는 것을 강조한다. 실험적으로, user scribble tool을 이용하여 user defined color & shape을 control해준 결과, object를 realistic하고 multi-view consistent하게 reconstruction할 수 있음을 보였다.

![motivation](/assets/posts/3d-rendering/editnerf/motivation.PNG)

Color와 shape에 대한 editing 뿐만 아니라 color/shape transfer도 가능하도록 모델 구성을 하였다.

# Related Works

- 본 editing task를 수행할 수 있도록 만들어주기 위해서는 sparse 2D user edits를 3D space 상에 잘 전달할 수 있어야 할 뿐만아니라, color와 shape 각각의 task를 담당하는 parameters을 효율적으로 학습시킬 수 있어야 한다고 주장한다. 기존의 neural scene representation networks들은 static objects에 대해 focus를 맞춰왔다고 논문에서 이야기하고 있다. 하지만, **본 논문에서는 효과적인 conditional training 방법을 위한 모델 구조 및 학습 방법을 제안했다.** 또한, 모델의 inductive bias에 집중하여 본 task를 수행할 수 있도록 한 것은 다른 volume rendering 연구와는 다르게 adversarial loss를 요구하지 않고도 consistent한 color와 shape을 할 수 있었다고 주장한다.

# Editing a Conditional Radiance Fields

![architecture](/assets/posts/3d-rendering/editnerf/architecture.PNG)

- **본 architecture는 크게 color를 담당하는 branch, shape을 담당하는 branch, shared network로 구성된다.** 저자는 여러 parameter updating 방법 (a), (b), (c)을 활용하여 test를 진행해보았다. 첫번째로, latent code만 업데이트하는 (a) 방법은 low-quality edit의 결과가 나왔다고 한다. (b) 방법은 전체 네트워크를 동시에 학습시키는 방법인데, 학습 속도가 굉장히 느렸고, unwanted changes가 보였다고 한다. 최종적으로 저자는 the later laters of the network 만 finetuing하는 (c) 방법을 택하였는데, computational cost도 줄일 수 있었고 edit하고자 하는 부분에 대해 잘 편집할 수 있었다고 한다. Task를 수행하기위한 입, 출력 식은 아래와 같다.

$$ (\textbf{c}, \sigma) = F(\textbf{x}, \textbf{d}, \textbf{z}^{(s)},\textbf{z}^{(c)}) $$

위 식의 특징은 기존 NeRF pipeline에서 color/shape code가 추가되었다는 점이다. 기존 NeRF-related 논문을 다뤄오면서 공통적으로 사용하였던 learnable parameter를 부여하여 표현력과 manipulation을 하고자 했던 의도로 보여진다.

이후, predicted color and density의 값으로 RGB rendering을 한 식은 아래와 같다.

$$ \hat{C}(\textbf{r}, \textbf{z}^{(s)}, \textbf{z}^{(c)}) = \sum_{i=1}^{N_c-1}c_i\alpha_iexp(-\sum_{j=1}^{i-1}\sigma_j\delta_j) $$

## Editing via Modular Network Updates

- 본 문단의 대부분의 내용은 위 architecture에서 설명하였고, efficient training scheme을 위해 부가적으로 제안한 **subsampling user constraints 와 feature caching**에 대해 간략히 설명하고자 한다.

- **subsampling user constraints**: 이미지 내의 모든 픽셀에 대한 ray에 대해서 처리하는 것이 아니라, 몇개만 sampling하여 처리하여 학습 속도 및 computational cost를 줄인 효과를 보였다.

- **feature caching**: 각각 color, shape edit에 대한 task를 수행하기 위해, 각 task에서 update가 필요하지 않은 부분은 weights를 고정하고 따로 caching을 해놓았다가, 이후에 필요할 때 사용한 방법이다. 이를 통해 불필요한 computation을 줄이며 rendering time 또한 대폭 감소할 수 있음을 보였다.

## Color Editing Loss

- 본 task에서는 object와 background를 구별하며 학습시키기 위하여 forground(object)와 background에 대한 mask가 user-predefined된 환경에서 editing을 수행한다. 추가적으로 바꾸고자 하는 target color에 대한 정보도 가지고 있다.

- 아래 식은 color editing시 reconstruction하는 loss 식을 나타낸 것이다.

$$ L_{rec} = \sum_{(\textbf{r}, \textbf{c}_f) \in y_f}^{}\left\| \hat{C}(\textbf{r}, \textbf{z}^{(s)}, \textbf{z}^{(c)}) - \textbf{c}_f\right\|+\sum_{(\textbf{r}, \textbf{c}_b) \in y_b}^{}\left\| \hat{C}(\textbf{r}, \textbf{z}^{(s)}, \textbf{z}^{(c)}) - \textbf{c}_b\right\| $$

- $c_f$는 forground, 바꾸고자하는 desired color를 동시에 의미하고, $c_b$는 background, unchanged가 되길 원하는 color를 의미한다.

- reconstruction loss를 기반으로 color editing에 필요한 total loss는 아래 식과 같다.

$$ L_{color} = L_{rec} + \gamma_{reg} \cdot L_{reg}$$

여기서 L_{reg}는 논문에 구체적인 수식으로 표현되어 있지는 않지만, large deviation을 방지하기 위해 original model와 updated model의 weight간 squared-distance를 좁혀주고자 했던 의도로 사용되었다. 추가적으로 color editing시에 latent code vector $\textbf{z}^{(c)}$와 $F_rad$를 같이 학습하고 $\gamma_{reg} =10$으로 설정했다고 한다.

## Shape Editing Loss

- 먼저 removal하는 task를 보면 color editing에서 로 density를 고려하는 term 하나가 추가되었다. 마찬가지로 foreground, background와 함께 지우고자 하는 location에도 user가 정의해준다.

$$L_{remove} = L_{rec} + \gamma_{dens}\cdot L_{dens} + \gamma_{reg}\cdot L_{reg}, \; \; L_{dens} = -\sum_{\textbf{r} \in y_f}^{}\sigma^\textbf{T}_\textbf{r}log(\sigma_\textbf{r})$$

- 흔히 알고 있는 Cross entropy loss 중의 하나로 Volume density가 1에 가까운 것은 1로, 0에 가까운 것은 0으로 만드는 loss function으로 shape removal을 수행하였다. 여기서는 $F_{dens}$, $F_{fuse}$ 모델을 학습하고 $\gamma_{dens} = 0.01$, $\gamma_{reg} = 10$으로 설정했다고 한다.

- Shape adding을 하는 training scheme은 다음과 같다. Original object와 target instance를 user interface에서 설정한 후에, desired paste location을 user가 설정해준다. 이후 각각 shape code, color code를 이용하여 editing 해준다. 수식은 아래와 같다.

$$ L_{add} = L_{rec} + \gamma_{reg} \cdot L_{reg} $$

- 여기서는 $F_{dens}$, $F_{fuse}$ 모델을 학습하고, $\gamma_{reg} = 10$으로 설정했다고 한다.

# Experiments

## Datasets

- Three public dataset을 활용하였다. (PhotoShape dataset, Aubry chairs dataset, GRAF CARLA dataset)

- 각각의 dataset들은 다른 instance, training-view를 가지고 있고, large apperance variation, large shape variation과 같은 특성을 가지고 있다.

#### Quantitative results

![result-table1](/assets/posts/3d-rendering/editnerf/result-table1.PNG)

- Table 1에서는 제안했던 모듈들에 대한 ablation study를 진행한 결과이다. 각 task를 담당하는 모델을 conditionally training했던 기법과 latent code 추가에 대한 성능이 좋아짐을 확인할 수 있었다. 또한 각 instance를 editing시, instance-level별로 쪼개어진 task를 하나의 전체 NeRF 모델을 사용했을 때 보다도 ours의 성능이 좋았다.

- Table 2에서는 color editing을 하기 위한 여러 approach들의 성능을 비교해보았을 때, ours의 성능이 높게 나옴을 알 수 있었다.

#### Qualitative results

![result-fig1](/assets/posts/3d-rendering/editnerf/result-fig1.PNG)

위 그림은 Color editing task에 대한 정성적인 결과를 내본 것이다. (c) 방법은 보이는 것과 같이 visual artifact가 많이 발생했고, (d)는 GAN 방법임에도 불구하고 unseen views들에 대해서는 잘 color edit을 잘 못하는 것을 볼 수있지만, ours는 color edit과 동시에 desired unchanged shape은 잘 유지함을 확인할 수 있었다.

![result-fig2](/assets/posts/3d-rendering/editnerf/result-fig2.PNG)

위 그림은 의자의 arm 부분을 remove하거나 빈 부분을 filling한 결과이다. 본 논문에서 제안한 여러 모듈들에 대해 ablation study를 해본 결과 인데, 각 모듈은 consistent shape을 잘 reconstruction할 수 있었고, whole network를 통째로 학습한 것보다 시간적으로나 성능적으로나 우월한 결과를 보였다.

![result-fig3](/assets/posts/3d-rendering/editnerf/result-fig3.PNG)

Source의 shape는 유지하며 target color로 바꾸거나, source color는 유지하며 shape을 바꾸는 실험을 진행해보았다. 각 task에 대해 바뀌어야할 부분은 잘 바뀌고, unchanged 되어야할 부분이 잘 남아있는 것을 확인할 수 있다.

![result-fig4](/assets/posts/3d-rendering/editnerf/result-fig4.PNG)

Real image에 대해 red color로 칠해보거나, shape editing을 한 결과이다. multi-view에 대해서도 consistent한 결과를 확인할 수 있었다.

# Conclusion

- 3D object의 category-level color/shape editing을 효과적, 효율적으로 volume rendering할 수 있는 방법을 제안했다. 구체적으로, color 혹은 shape를 담당하는 branch를 만드는 등 model configuration에 대한 조정이 있었고, 각 task를 담당하는 branch에서 conditionally parameter update를 한 것이 main contribution이다.

# Discussion

- 실험 demo 결과를 직관적으로 확인할 수 있도록 UI를 잘 만들어서 이목을 끌 수 있었던 것 같다.

- 모델을 task에 맡게 design했던 것이 굉장히 high-level하고 직관적인 느낌이었지만, 결과적으로 좋은 성능을 보였다는 점이 신기했다. Cherry picking일 수도 있겠지만, 독자가 궁금해할법한 실험 및 실험 결과들을 충분히 보여주었기 때문에 accept이 될 수 있지 않았을까라는 의견이다.

- 또한, 논문에서 계속해서 computation speed, training speed를 언급하는데, 이부분에서 지적을 받은 것 같다. 아무래도 instance-level semantic 정보를 아무런 semantic supervision없이 control하는 것은 사실 상 불가능하다고 생각이 드는데, 방대한 양의 데이터 셋 구성 자체가 instance-level로 편집이 되어있는 것을 사용하기에 모델이 알아서 잘 add, remove할 수 있지 않았을까라는 생각이 들었다.
