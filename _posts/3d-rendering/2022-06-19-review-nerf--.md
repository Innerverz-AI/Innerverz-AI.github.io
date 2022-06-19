---
layout: post
title:  "[Paper Review] NeRF−−: Neural Radiance Fields Without Known Camera Parameters"
author: 조우영
categories: [Deep Learning, Multiple-View, Volume Rendering, Neural Radiance Fields, Camera parameters]
image: assets/images/logo-3d-rendering.jpeg
---

글을 읽기 전에, 본 글은 vanila NeRF에 대한 사전 지식을 알고 있는 독자들을 대상으로한 글이므로, 본 블로그의 NeRF 리뷰를 먼저 읽기를 추천한다.

# Contributions
- 새로운 view point에서 관찰한 이미지를 합성시키는 task (Novel View Synthesis: NVS)에서 NeRF는 각 이미지에 대한 위치 정보가 필요하였다. 하지만 본 논문에서는 pre-computed parameter 없이 NVS가 가능하고 기존 vanila NeRF와 comparable한 성능을 보였다.  
- NeRF 모델과 camera parameters이 jointly training되는 학습 기법을 제안했다.

![motivation](/assets/posts/3d-rendering/nerf--/motivation.PNG)

# Related Works

- 기존 NeRF는 NVS를 수행하기 위해 COLMAP이라는 라이브러리를 사용하여 camera parameter를 구하는 전처리를 수행한다. 하지만 전처리 과정에서 COLMAP은 dynamic & complex scene에 대해서 완벽히 동작하지 못하므로, NeRF로 하여금 COLMAP pre-computed parameters에 training dependent 시키는 것은 모델의 안정성에 영향을 미칠 수 있다.
- 따라서, NVS를 수행하며 NeRF 모델과 camera parameters를 jointly trainnig해주어야할 필요성이 있다고 저자는 제안한다.  

# Nerual Radiance Field Without Known Camera Parameters

- 기존 NeRF의 동작을 수식으로 표현하자면 아래 식과 같다.

$$ \hat{I}_i(\textbf{p}) = \mathcal{R}(\textbf{p}, \pi_i|\Theta) = \int_{h_n}^{h_f}T(h)\sigma(\textbf{r}(h))\textbf{c}(\textbf{r}(h), \textbf{d})dh, \; where \ T(h)= exp(-\int_{h_n}^{h}\sigma(\textbf{r}(s)) \ ds) $$

여기서 Loss는 NVS되는 새로운 이미지의 rgb값과 같아지도록 학습되고, 학습되는 parameter에 대한 수식은 다음과 같다.

$$\mathcal{L} = \sum_{i}^{N}||I_i-\hat{I}_i||^2_2 \; \;\;  \; \Theta^* = arg \ \underset{\Theta}{min}\mathcal{L}(\hat{I}|I, \Pi)$$

기존 NeRF와는 다르게 본 논문에서는 NeRF 모델의 parameter와 camera parameterㄹ를 jointly training해보겠다라는 의도로 아래 수식과 같다.

$$\Theta^*, \Pi^* = arg \ \underset{\Theta, \Pi}{min}\mathcal{L}(\hat{I}, \hat{\Pi}|I)$$

이때 $\Pi$는 camera intrinsic & extrinsic parameter를 모두 포함한다.

위 내용의 basics를 이해하고, NeRF-- method의 쉬운 이해를 돕기 위해 overall pipeline을 먼저 알아보고자 한다.

## NeRF-- pipeline

![overall-pipeline](/assets/posts/3d-rendering/nerf--/overall-pipeline.PNG)

### Step 1: Initialization of camera parameters

- 먼저 camera parameters들을 학습가능한 tensor로 정의하고, 각각의 parameter들은 다음과 같은 초기값을 가진다. $\hat{f_x}, \hat{f_y}: FOV \cong 53$ (per image sizes) $(h, w)$, $\hat{t_i}$ with zero vector, $\hat{\phi_i}$ with identity matrix.

### Step 2: Ray Construction
- 이미지의 spatial pixel 정보와 여러 camera parameter를 이용하여 각 이미지 pixel에 해당하는 ray를 모델링할 수 있고, 아래 수식이 그에 해당한다.

$$\hat{\textbf{r}}_{i,m}(h) = \hat{\textbf{o}}_i + h\hat{\textbf{d}}_{i,m}, \; \hat{\textbf{d}}_{i,m} = \hat{\textbf{R}}_i
\begin{pmatrix}
(u-W/2)/\hat{f}_x) \\
-(v-H/2)/\hat{f}_y)\\
-1
\end{pmatrix}

//

\textbf{R} = \textbf{I} + \frac{\sin(\alpha)}{\alpha}\phi^\wedge  + \frac{1-\cos(\alpha)}{\alpha^2}, \; \phi^\wedge = \begin{pmatrix}
\phi_0 \\
\phi_1 \\
\phi_2
\end{pmatrix}^{\wedge} =

\begin{pmatrix}
0 & -\phi_2 & \phi_1 \\
\phi_2 & 0 & -\phi_0 \\
-\phi_1 & \phi_0 & 0 \\
\end{pmatrix}
$$

보통 3D rotation representation을 할 때, Euler coordinate을 많이 사용하지만, 여기서는 한 고정 point에서 rotation 정보를 표현하기에 유용한 the axis-angle representation $\phi := \alpha \mathbf{\omega}, \; \phi \in \mathbb{R}^3$을 활용한 것이 특징적이다.

### Step 3: Jointly Optimisation of NeRF and Camera parameters

![joint-opt](/assets/posts/3d-rendering/nerf--/joint-opt.PNG)

- NeRF 모델과 camera parameters들은 위 수식의 pre-defined constraint와 함께 학습되고, color:$c$와 density: $\sigma$를 내뱉고 이를 이용해 color rendering을 통해 jointly training된다.  

### Additional step: Refinement

- 위와 같이 camera parameter와 NeRF model의 parameter를 jointly 학습하는 경우에는 local minimum에 빠질 확률이 높다고 저자는 서술하고 있다. 이 문제를 완화하기 위해 저자는 이전 step의 pre-trained camera parameter를 저장해놓고, NeRF 모델을 초기화한 후에, 다시 jointly optimization하는 기법을 활용하고 있다.


# Experiments

## Datasets

### LLFF-NeRF dataset

- 8 forward-facing scene captured by mobile phones containing 20-62 images
- 756 x 1008 resolutions and ebery 8th image is used as the test image.

### RealEstate 10K, Tanks & Temples dataset

- various resolution between 480x460 ~ 1080x1920
- frame rates 24 fps to 60 fps


#### Quantitative results

![result-table1](/assets/posts/3d-rendering/nerf--/result-table1.PNG)

여러 scene에 대해서 COLMAP + NeRF 와 NeRF--의 NVS 성능을 비교해본 표이다. 이미지의 quality를 나타낼 수 있는 PSNR, SSIM, LPIPS 지표를 사용하여 평가해보았다. 특정 scene (Orchids, Room)에서 COLMAP에 비해 떨어지는 성능을 보이기도 했으나, 전반적으로 좋은 성능이 나옴을 강조하고 있다. 또한 refinement에 대한 성능 향상 효과도 보이고 있다.

![result-table2](/assets/posts/3d-rendering/nerf--/result-table2.PNG)

LLFF-NeRF dataset에 대해 각 trainable parameter의 configuration을 변형해가며 실험한 table이다. 실험 1 (E1)에서는 NeRF-- 통해 얻은 focal length, camera pose와 COLMAP estimated pose와의 오차가 작게 나오는 것을 말한다. 이후 E2-E3에 대해 config 요소를 제외하여 실험해보아도 여전히 여러 요소에서 오차값이 적게 나옴을 강조하고 있다.

#### Qualitative results

![result-fig1](/assets/posts/3d-rendering/nerf--/result-fig1.PNG)

Baseline: COLAMP + NeRF와 정성적으로 비교해본 실험으로 detail적 측면에서 NeRF--의 성능이 더 좋게 나옴을 말하고 있다.

![result-fig2](/assets/posts/3d-rendering/nerf--/result-fig2.PNG)

학습되는 epoch에 따라 COLMAP의 camera trajectory를 NeRF--가 학습해 나감을 보이고 있다. NeRF 이미지들을 일반적으로 Monocular camera로 부터 얻은 이미지들 이고, scale에 대한 보정을 해준 metric ATE aligned를 활용해보면, COLMAP이 구한 경로와 거의 일치함을 알 수 있다.

![result-fig3](/assets/posts/3d-rendering/nerf--/result-fig3.PNG)

baseline NeRF는 때때로 real scene에 대해 failure case가 나오게 되는데, COLMAP은 (c)와 같이 NVS에 실패하였고, 이를 수동적인 보정을 통해 얻은 이미지가 (d)
이다. (d)의 rgb이미지는 NeRF--에 비해 blurry한 모습을 보이고 depth map은 아얘 잘 표현하지 못한반면, NeRF--의 결과 (b)는 rgb, depth 이미지가 잘 표현되고 있음을 알 수 있다.

![result-fig4](/assets/posts/3d-rendering/nerf--/result-fig4.PNG)

COLMAP은 때때로 abnormal camera trajectory를 그리는 반면, NeRF--는 더욱 smooth한 경로를 예측함을 알 수 있다.  

![result-fig5](/assets/posts/3d-rendering/nerf--/result-fig5.PNG)

COLMAP은 camera pose prediction에 실패한 경우이고, NeRF--는 잘 표현하는 example이다.

# Conclusion

- Camera parmeter를 활용하여 ray를 기하학적으로 모델링 하였고, 이의 관계 및 NeRF 모델을 jointly training하여 pre-computed camera parameter 없이도 NVS를 할 수 있음을 보였다.
- 때때로 dynamic, complex scene에서 잘 동작하지 않는 COLMAP 라이브러리의 성능을 뛰어넘는 결과를 보였다.


# Discussion

camera parameter의 의존도를 떨어뜨림과 동시에 NeRF를 학습 및 NVS를 수행할 수 있었지만, 결국 자신이 정확히 원하는 위치 및 카메라 정보에 대한 이미지를 찾을 수 없다는 단점이 있다. 물론 랜덤 포인트를 많이 sampling하여 근접한 두 sample간의 interpolated 정보는 찾을 수 있겠지만, 이는 정확하지 않다.

과연 GAN Inversion의 연구가 활발했던 것처럼, 새로운 view point의 이미지에 대한 카메라 위치 정보 및 각도를 inversion해서 찾아내는 연구는 필요한 연구일까? 라는 생각도 해본다.
