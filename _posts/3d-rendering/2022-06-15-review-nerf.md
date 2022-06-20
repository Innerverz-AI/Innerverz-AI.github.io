---
layout: post
title:  "[Paper Review] NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
author: 조우영
categories: [Deep Learning, Multiple-View, Volume Rendering, Neural Radiance Fields]
image: assets/images/logo-3d-rendering.jpeg
---

글을 읽기 전에, 3D 카메라 좌표계와 3D 물체가 카메라에 맺히는 원리(camera projection)를 이해하는 것을 추천한다.

# Contributions
- 카메라의 위치 및 각도 정보를 활용하여 **새로운 카메라의 위치 및 각도에서 관찰한 이미지**를 생성해낼 수 있다.
- Multi Layer Perceptron을 활용한 단순한 구조이다.
- NeRF의 representation을 활용해서 입력의 표현력이 부족하였다. 이를 개선하기 위해 **positional encoding**을 도입했다.
- Sampling시 요구되는 sample point 및 query 수를 줄이기 위하여 **hierarchical sampling procedure**를 제안한다.

![motivation](/assets/posts/3d-rendering/nerf/motivation.png)

# Related Works

3D 물체의 realistic scene representation을 표현하기 위한 기존 방법은 triangle meshes 혹은 voxel grid가 활용되어왔다. 하지만 위와 같은 discrete methods는 scene에 포함되어 있는 complex geometry를 고해상도 이미지로 표현하기에는 역부족이었지만, 본 방법의 differential volumentric representation approach는 기존 방법의 결과에 대한 한계를 극복할 수 있었다고 논문에서 서술한다.


# Nerual Radiance Field Scene Representation

- NeRF는 3D location $x = (x, y,z)$와 2D viewing direction $d = (\theta, \phi)$를 입력받아 emitted color $c = (r, g, b)$와 volume density $\sigma$ (투명도의 역수 개념을) 내뱉는 vector valued function으로 연속적인 장면을 표현하고자 한다.
- 연속적인 5D scene representation을 MLP network $F_\Theta:(x,d)\rightarrow(c,\sigma)$로 모델링한다.
- $\sigma$는 location $x$를 입력받아 예측되고, color $c$는 location $x$와 viewing direction $d$를 모두 활용하여 예측된다.

![overall-framework](/assets/posts/3d-rendering/nerf/overall-framework.png)

## Architecture

- 3D coordinate $x$가 8 layer MLP를 통과하여 $\sigma$와 256차원 feature vector를 출력한다.
- Feature vector와 viewing direction을 concat하여 1 layer MLP를 통과시켜 color를 얻는다.

![architecture](/assets/posts/3d-rendering/nerf/architecture.png)

# Volume Rendering with Radiance Fields

NeRF에서 활용되는 volume rendering에 대한 수식은 아래와 같이 정의된다.

$$ C(\textbf{r}) = \int_{t_n}^{t_f} T(t)\sigma(\textbf{r}(t)), \textbf{d})dt, \; where \ T(t)=exp(-\int_{t_n}^{t})\sigma(\textbf{r}(s))ds$$

- 논문의 표현을 인용하면, volume density $\sigma(x)$는 location $x$의 극소점에서 끝나는 ray의 확률의 미분값으로 이해할 수 있다.
- 쉽게는 density가 클수록 weight가 커야하며 어떤 지점을 가로막고 있는 점들의 density의 합이 작을 수록 weight가 커야한다.
- $T(t)$는 ray를 따라서 $t_n$에서부터 $t$까지의 누적된 투과도이다. (ray가 $t_n$에서 $t$까지 이동하는 동안 어떤 입자와도 부딪히지 않을 확률과 같다.)
- **계층화된 샘플링 방식**을 사용한다. $[t_n, t_f]$를 N개의 구간으로 나누고 각 구간에서 uniform 샘플링했다.

$$ t_i \sim \mathcal{U}[t_n + \frac{i -1}{N}(t_f-t_n), t_n + \frac{i}{N}(t_f-t_n)] $$

- 위 샘플링 방식을 이용하여 continuous space를 discrete space로 응용하여 rendering 식을 정의하면 아래와 같다.

$$ \hat{C}(\textbf{r})=\sum_{i=1}^{N}T_i(1-exp(-\sigma_i\delta_i)\textbf{c}_i, \; where T_i = exp(-\sum_{j=1}^{i-1}\sigma_i\delta_i) $$

# Optimizing a Neural Radiance Field

-  앞서 서술한 방법들만으로는 SOTA 성능을 얻기에는 불충분했다.
- 따라서 high resolution complex scene을 표현하는데 도움이 되는 두가지 요소를 추가했다.

## Positional encoding


- 위와 같은 volume rendering 방법만으로는 color와 geometry의 high-frequency variation을 표현하는데 취약했다.

- 5D로 입력되는 차원 좀 더 높여서 표현력을 가질수 있도록 high dimension으로 만들어주는 행위이다. $\gamma$로는 아래의 수식을 사용했다.

$$ \gamma(p) = (\sin(2^0\pi p), \cos(2^0\pi p), \cdots , \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p)) $$

![result-fig0](/assets/posts/3d-rendering/nerf/result-fig0.png)

## Hierarchical volume sampling

- Ray선 상에서 샘플링을 할 때, 랜더링된 이미지에 기여하지 않는 free space나 가려진 부분에서도 반복해서 샘플링된다는 문제가 있었다.
- 따라서 동시에 2개의 네트워크를 최적화하는 hierarchical representation을 제안한다.
- $N_c$ location의 집합을 샘플링해서 coarse network를 evaluate한다.
- Coarse network의 output을 이용해 volume에 연관된 지역에서 더 샘플링한다. 본 샘플링 방식을 수식으로 나타내면 아래와 같다.

$$ \hat{C}_c(\textbf{r}) = \sum_{i=1}^{N_c}w_ic_i, \; \; w_i = T_i(1-exp(-\sigma_i\delta_i)) $$


- Weights에 해당하는 w를 normalize하면 확률변수로 간주할 수 있다.
- 이 확률 분포로 부터 $N_f$ locations의 집합을 샘플링하고 첫번째 집합과 두번째 집합 모두를 이용하여 fine network를 evaluate하고 최종 랜더링을 한다.

# implementaion details

- 하나의 scene에 대해 Nerual Network를 최적화하였다.
- 촬영된 scene의 RGB 이미지, camera pose, intrinsic parameter, scene bound를 포함하는 데이터셋이 필요하다. (synthetic data에서는 gt를 이용했고 real data는 이 값을 추정하기 위한 라이브러리를 사용했다.)
- Total loss function은 아래와 같다.

$$ L = \sum_{r\in \mathbb{R}}^{}[\left\|\hat{C}_c(\textbf{r}) - C(\textbf{r}) \right\|^2_2 + \left\|\hat{C}_f(\textbf{r}) - C(\textbf{r}) \right\|^2_2] $$

- Batch size: 4096개의 ray를 활용했다.

# Experiments

## Datasets

### Synthetic rendering of objects

- Diffuse Synthetic 360 (4 Lambertian objects, 479 input 1000 test)
- Realistic Synthetic 360 (proposed in this work, 8 objects, 100 input 200 test)

### Real images of complex scenes

- Real Forward-Facing (8 scenes captured with a handheld cellphone, 20 to 62 images)

## Comparison

### 비교 모델: Neural Volumes, Scene Representation Networks, Local Light Field Fusion

#### Quantitative results

![result-table1](/assets/posts/3d-rendering/nerf/result-table1.png)

여러 Synthetic & real dataset에 대해 PSNR, SSIM, LPIPS의 metric으로 평가해본 결과이다. 각 Dataset은 geometrically complex scenes를 포함하고 있다. Real Forward-Facing scene에서는 LLFF method의 LPIPS 성능이 다소 높게 나왔으나, 정성적인 결과는 NeRF가 더 우수한 것을 확인할 수 있었다.

#### Qualitative results

![result-fig1](/assets/posts/3d-rendering/nerf/result-fig1.png)
![result-fig2](/assets/posts/3d-rendering/nerf/result-fig2.png)

Synthetic dataset에 있는 각 objects에 대해 novel view synthesis를 해본 결과 이미지이다. 저자는 LLFF나 SRN 방법을 활용한 결과에 대해 다소 blurry, banding artifacts, ghosting artifacts등 high quality image를 합성하기는 부족했다고 평가한다. 그러나, NeRF의 방법으로는 ground truth와 근접한 이미지를 만들어 낼 수 있음을 그림을 통해 보였다.

![result-fig3](/assets/posts/3d-rendering/nerf/result-fig3.png)
![result-fig4](/assets/posts/3d-rendering/nerf/result-fig4.png)

Real world scenes가 포함된 Dataset에 대해 novel view synthesis를 해본 결과 이미지이다. 저자는 LLFF, SRN 방법을 활용한 결과 이미지에 대한 not clean, repeated edges, not fine detail함을 지적하고 있다.

### Ablation studies

![result-table2](/assets/posts/3d-rendering/nerf/result-table2.png)

1) ~ 9) case까지 여러 Hyperparameters와 실험 세팅을 조절하며 정량적인 평가를 내린 표이다. 가로축 차례대로, 입력 차원, 이미지 개수, Positional Encoding Frequency 파라미터, 샘플링 파라미터를 의미하고, 세로축으로는 각각 실험한 세팅 환경을 의미한다. 제안한 모든 요소를 포함시켰을 때, 성능이 가장 뛰어난 것을 확인할 수 있다.

# Conclusion

- 5D 입력과 MLP를 활용하여 기존 volume representation approach들 보다 좋은 성능을 얻을 수 있었다.
- Efficient and effective한 샘플링을 수행하기 위하여 hierarchical sampling방법을 제안하였다.
- 입력 dimension을 높이기 위한 positional encoding 방법이 사용되었고, 실제 결과를 비교했을 때 효과적이었다.
- 앞으로 더욱 효율적인 샘플링 방법과 interpretability의 연구 방향에 대한 가능성이 유망해보인다.

# Discussion

NeRF가 3D 공간 정보를 담을 수 있는 새로운 문을 열었다고 생각한다. 그렇다면, 후속 논문들을 읽을 때 다음 항목에 주안점을 두며 읽어나가야 할 것이다.

- 다양한 application에서 NeRF가 어떻게 활용되었는지 ?
- 어떻게 NeRF가 3d geometry에 대한 정보를 잘 담아낼 수 있게 되었는지 ? 이 과정 중에 NeRF가 다루지 못했던 한계점이 있을지 ?
