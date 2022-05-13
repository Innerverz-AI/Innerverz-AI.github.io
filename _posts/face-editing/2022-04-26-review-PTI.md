---
layout: post
title:  "[Paper Review] PTI: Pivotal Tuning for Latent-based Editing of Real Images"
author: 류원종
categories: [Deep Learning, GAN Inversion, PTI, Latent Manipulation]
image: assets/images/logo-face-editing.jpeg
---

![title](/assets/posts/face-editing/2022-04-26-review-PTI/title.PNG){: width="100%", height="100%"}<br>

# Motivation
Real image editing 을 위해서는 reconstruction 과 editing 둘 다 잘되어야 하는데 좋은 방법 없을까?

# Summary
1. Generator 를 주어진 real image 에 finetuning 해서 reconstruction 성능을 끌어올린다. 

2. Distortion-editability trade-off 를 극복함으로써 editing 성능도 놓치지 않는다. 

3. Real imag editing 이 가능하며, multi images 도 pivotal tuning 가능하다.

# Backgrounds
- StyleGAN 의 latent space 는 이미 disentangled 되어 있어 editing capability 가 높다. 이런 특성을 이용해 age, expression, rotation 등을 조절하는 방법들이 제안되었다. (latent manipulation)

- 랜덤으로 샘플링 된 이미지는 편집이 용이했지만, 새롭게 주어진 이미지(real image)를 편집하기 위해서는 우선 이미지를 latent space 상에 projection 해야 한다. (GAN inversion) 

- StyleGAN 의 W space 의 차원은 $R^{512}$  이지만, Image2StyleGAN 에서 제안한 W+ space 는 $R^{18 \times 512}$ 차원이라 표현력이 더 좋다. 

- W+ space 는 out-of-domain image 까지 잘 표현할 수 있었다. 하지만 W+ space 에 속한 latent vector 는 StyleGAN 의 W space 에서 벗어나 있었기 때문에, editing 성능이 떨어진다. 

- 같은 W+ space 를 이용하더라도, encoder-based GAN inversion 보다 optimization-based GAN inversion 의 reconstruction 성능이 더 좋다. 하지만 editing 능력은 전자가 후자보다 더 낫다.

> Real image 에 대해 reconstruction 이 잘 되는 w vector 를 찾을수록 editing 능력이 떨어지는 현상을 보인다. 이 문제를 distortion-editability trade-off 라고 부른다. W+ space 의 latent vetor 중에서도 W space 에 가까운 것일수록 editability 가 좋다. e4e 에서는 W+ latent space 의 latent vector 중에서도 W space 에 근접한 것들을 찾아내 reconstruction 과 editing 성능을 동시에 잡으려 했고, 이런 latent vector 를 sweetie-spot 라고 했다. 


## Inversion
- Inversion step 의 목적은 real image $x$ 에 대응하는 $w_p$ 를 찾는 것이다. 

- 아래와 같은 Loss function 을 이용해 latent vector $w$ 를 optimization 한다. 즉 generator 에 $w$ 를 입력했을 때 $x$ 를 잘 복원하도록, $w$ 를 조금씩 수정하는 과정을 반복한다.

$$
w_p, n=\underset{w, n}{\operatorname{argmin}} \ L_{LPIPS}(x, G(w, n; \theta))+\lambda_nL_n(n))
$$

- 이때 찾아낸 $w_p$ 를 StyleGAN 에 입력해 만든 이미지는 주어진 이미지와 유사하지만 완벽히 똑같진 않다. 

$$
x \approx G(w_p;\theta)
$$

- 여기서 찾은 $w_p$를 pivotal tuning 의 기준점으로 사용한다. 

- Details
  - 이 단계에서 generator 는 고정되어 있다.
  - mapping network (Z → W)는 사용하지 않고, synthesis network (W → I) 만 사용한다.
  - W space 의 editability 가 가장 좋으므로 W space 를 타겟으로 한다.
  - noise regularization 을 이용해 noise vector 에 중요한 정보가 포함되지 않도록 한다.

## Pivotal tuning
### Baseline
- 위에서 얻은 $w_p$ 를 pivot 으로 고정시키고 generator $G$ 를 finetuning 한다.

- Generator’s weight $\theta$ 를 학습한 결과를 $\theta^*$ 라고 할 때, $w_p$ 를 입력해서 얻은 이미지를 $x^p$ 라고 한다. 

$$
x^p=G(w_p;\theta^*)
$$

- $\theta^*$ 은 아래와 같은 loss function 으로 optimization 해서 얻을 수 있다.  

  - Loss function: 
  $ L_{pt} = L_{LPIPS}(x, x^p) + \lambda_{L2}L_{L2}(x, x^p) $
  - Optimization: $ \theta^{\*}=\underset{\theta^*}{\operatorname{argmin}} \ L_{pt} $

- Details 
  - 이 단계에서 $w_p$ 는 고정되어 있다.

> 
이를 N 개의 이미지에 대한 Pivotal tuning 으로 확장하면 아래와 같이 쓸 수 있다.
\$$
L_{pt} = \dfrac{1}{N} \sum_{i=1}^{N}(L_{LPIPS}(x_i, x_i^p) + \lambda_{L2}L_{L2}(x_i, x_i^p))
\$$

### Additional loss term: Locality Regularzation
![regularization](/assets//posts/face-editing/2022-04-26-review-PTI/regularization.PNG){: width="100%", height="100%"}<br>

- Pivotal tuning 결과, $w_p$ 와 멀리 떨어진 non-local latent code 로 생성한 이미지들은 퀄리티가 나빴다.

$$
G(w_{random};\theta) \ \ \ \ \neq \ \ \ \ G(w_{random};\theta^*)
$$

- PTI 가 styleGAN latent space 에 미치는 영향을 줄이기 위해, regularization term 을 추가해 PTI의 영향을 $w_p$ 근처의 local region 으로 제한한다. 

- $w_z = mapping(z_{random})$ 와 pivotal latent code $w_p$ 를 interpolation 해서 $w_r$ 를 얻는다. 

$$
w_r = w_p + \alpha \dfrac{w_z-w_p}{||w_z-w_p||_2}
$$

- $x_r = G(w_r;\theta)$ 와 $x_r^{\*} = G(w_r;\theta^{\*})$ 의 차이를 최소화 하도록 loss function 을 아래와 같이 설정한다. $w_p$ 주변의 $w_r$ 에서 $G(w_r;\theta) = G(w_r;\theta^*)$ 를 만족한다면, 그 바운더리를 벗어나는 영역엔 영향이 없다고 볼 수 있다.

$$
L_{R} = L_{LPIPS}(x_r, x_r^*) + \lambda_{L2}^RL_{L2}(x_r, x_r^*)
$$

![vector](/assets//posts/face-editing/2022-04-26-review-PTI/vector.png){: width="100%", height="100%"}

> 어차피 real image 를 복원하고 편집하기 위해 PTI 를 사용하는데 왜 random sampled image 의 퀄리티까지 고려할까? PTI 과정 중에 latent space 가 변형되면 latent manipulation 을 위한 semantic direction vector 의 유효성이 떨어지기 떄문인거 같다.

- 그리고 이를 $N_r$ 개 random latent codes 로 확장한다.

$$
L_{R} = \dfrac{1}{N} \sum_{i=1}^{N}(L_{LPIPS}(x_{r,i}, x_{r,i}^*) + \lambda_{L2}L_{L2}(x_{r,i}, x_{r,i}^*))
$$

- 최종적으로, 아래와 같이 Generator's optimization function 에 locality regularization 을 추가한다. 

$$
\theta^*=\underset{\theta^*}{\operatorname{argmin}} \ L_{pt}+\lambda_RL_R
$$

# Result

- Encoder-based 방식(e4e)과 opimization-based 방식(SG2, SG2 W+)에 비해 detail 표현 능력이 뛰어나다.

![recon](/assets//posts/face-editing/2022-04-26-review-PTI/recon.PNG){: width="100%", height="100%"}<br>

- 다른 방법들에 비해 editing 능력도 뛰어나다 (+smile, +age, -beard)

![editing](/assets//posts/face-editing/2022-04-26-review-PTI/editing.PNG){: width="100%", height="100%"}<br>

# Conclusion
- Short forward pass 를 위해서 PTI 를 근사하는 trainable mapper 가 필요함
- individual 에 대해 이미지 한 장이 아닌 이미지셋으로 학습하면 더 좋을듯
- StyleGAN 이 아닌, 다른 Generator 에도 적용해 볼 수 있음 
- 모델 캐스팅에 유용하게 쓰일 수 있음
