---
layout: post
title:  "[Paper Review] Fast, Robust, Controllable Text to Speech"
author: 김재훈 
categories: [Deep Learning, Text-to-Speech, Speech Synthesis, Transformer]
image: assets/images/logo-speech.jpeg
---

# 1. Overview

Autoregressive 방식을 사용한 기존 TTS 모델들과 다른 Feed-Forward Transformer을 사용해 mel spectrogram을 parallel하게 생성한다. 또한 duration prediction을 통해 단어가 생략되거나 반복되는 오류를 삭제할 뿐만 아니라 자연스러운 말투나 억양을 담당하는 prosody에 더 robust한 결과를 도출한다. Non-autoregressive 특성 상 속도가 아주 빠르고 자연스러운 목소리 데이터를 출력하며 length regulator 파라미터 변경을 통해 말하는 속도도 조절이 가능하다.

# 2. Methods

![Screen Shot 2022-06-20 at 7.26.38 PM.png](/assets/posts/TTS/fastspeech/Screen_Shot_2022-06-20_at_7.26.38_PM.png)

## 2-1. Feed-Forward Transformer

Transformer는 FastSpeech가 적은 파라미터와 빠른 속도를 이룰 수 있는 가장 큰 이유이다. 목소리 데이터는 continuous하기 때문에 인접하거나 이전의 데이터가 중요하고, 이로 인해 RNN, autoregressive 모델들이 좋은 퍼포먼스를 보였지만, 모두 이전 연산을 기다려야 하기 때문에 속도가 매우 느린 단점이 있었다. Transformer을 활용함으로써 필요한 정보를 효과적으로 사용하고 병렬 연산으로 속도 또한 크게 개선했다.

논문에서는 multi-head self attention과 1D convolution layer을 함께 사용하는 Feed-Forward Transformer 모듈을 제시하고, 음소(phoneme) 분석과 mel spectrogram 생성단계에서 *N*개 FFT 모듈을 sequential하게 사용한다 (논문에서 학습한 모델의 경우 6).

기존 transformer들이 2개의 dense layer을 사용한 부분에서 FFT는 1D convolution을 사용했는데, 목소리의 특성 상 인접한 hidden state 값들이 중요한 영향을 끼치는 것을 반영하기 위함이다.

## 2-2. Length Regulator

텍스트에서 스펙트로그램으로 변환할 시, 발음 별로 스펙트로그램의 길이를 정해주는 Length Regulator이 사용된다. 전처리를 통해 단어와 글자들은 발음의 기본 단위를 나타내는 음소 단위로 쪼개져 모델 입력으로 들어가지만, 출력으로 나오는 mel spectrogram의 temporal dimension은 이보다 훨씬 길다. ‘ㅏ'라는 발음을 0.3초간 발음한다고 생각했을 때, 음소는 하나이지만 프레임당 0.1초를 나타내는 스펙트로그램은 3 프레임이 생성되어야 한다. 음소마다 발음의 시간이 다르고 미묘한 발음의 차이를 담기 위해서는 모든 입력 텍스트에 대해 출력하는 스펙트로그램의 길이를 예측해야 하는데, 이를 Length Regulator을 통해 진행한다.

$$\mathcal{H}_{mel}=\mathcal{LR}(\mathcal{H_{pho}}, \mathcal{D}, \alpha)$$

H_mel, H_pho는 각각 스펙트로그램과 음소의 hidden state을 가리킨다. D는 duration sequence로 각 음소마다 얼마나 긴 스펙트로그램이 필요한지 나타낸다. alpha값은 expansion을 얼마나 줄지에 대한 hapram이다 (기본은 1, 빠른 목소리는 1보다 작고 느린 목소리는 1보다 큰 값으로 설정).

예를 들어 ‘김' 이라는 단어가 입력으로 들어왔을 때, 음소로 쪼개진 시퀀스는 [’ㄱ', ‘ㅣ', ‘ㅁ'] 과 같이 전처리 되어 입력으로 들어간다. 음소들이 각각 [3, 2, 5] (duration sequence)개의 스펙트로그램 프레임을 만들어야 한다면 H_mel은 [’ㄱ', 'ㄱ', ‘ㄱ', ‘ㅣ', ‘ㅣ', ‘ㅁ', ‘ㅁ', ‘ㅁ’, ‘ㅁ', ‘ㅁ']이 된다. alpha 값이 0.5가 된다면 duration seuqnece에 스칼라 곱과 반올림으로 [2, 1, 3]으로 변환된다.

## 2-3. Duration Predictor

Duration sequence가 있다면 입력 문장에 대해 출력되는 스펙트로그램 길이를 정확하게 맞춰 같은 말을 반복하거나 스킵하는 경우를 크게 줄일 수 있다. 하지만 음소마다 발음되는 길이는 어떻게 정하는가? 

발음은 사람마다 모두 다르기 때문에 고정된 key-value pair을 가질 수 없다. 학습 데이터에 대한 정확한 duration mapping을 얻기 위해 다른 pretrained TTS 모델을 선생님으로 두고 정보를 추출해 온다.

Autoregressive Transformer TTS 모델은 좋은 성능을 내지만 속도가 느린 단점이 있는데, 이 모델을 사용해 phoneme마다 duration 정보를 가져온다. pretrained 모델에 텍스트를 입력했을 때 transformer에서 연산되는 다양한 attention matrix를 얻을 수 있는데, 여기서 focus rate을 계산한다.

$$\mathcal{F=\frac{1}{S}\sum_{s=1}^{s=S} \textbf{max}_{1\leq t\leq \text{T}}(a_{s,t})}$$

focus rate에서 S는 스펙트로그램, T는 텍스트 (음소), a는 attention matrix에서 (s, t) 포지션에 있는 값을 가리킨다. transformer에서 attention은 alignment를 도와주는 역할을 하지만 모든 attention이 diagnonal하게 align되어있지는 않다. 음소-스펙트로그램 관계를 매핑하는 attention을 찾기 위해 focus rate 값이 사용되며, 모든 attention에서 F를 구한 후 값이 가장 높은 attention을 duration에 대한 레퍼런스로 사용한다.

![images.png](/assets/posts/TTS/fastspeech/images.png)

위와 같이 음소-스펙트로그램을 잘 매핑하는 attention이 선정되면 각 음소마다 몇 프레임의 스펙트로그램이 필요한지 계산한다. 위와 같이 한 음소마다 한 프레임 이상이 할당되고, 이 정보는 추합되어 최종적으로 duration sequence가 완성된다.

$$\mathcal{d_i=\sum_{s=1}^{S}\textbf{argmax}_t(a_{s,t}=i)}$$

Duration sequence의 각 element는 i번째 음소에 해당하는 attention energy가 max값을 가지는 포지션의 개수가 되며, 다시 말해 음소가 차지하는 스펙트로그램 프레임의 개수이다.

# 3. Results

결과에 대한 평가는 다른 TTS 모델들과 같이 정량적보다는 정성적 결과가 주를 이룬다. 해당 논문은 대표적으로 사용되는 Mean Opinion Score (MOS)와 ablation study에서 비교를 목적으로 Comparative MOS (CMOS)가 사용되었다.

## 3-1. Audio Quality

![Screen Shot 2022-06-21 at 4.37.52 PM.png](/assets/posts/TTS/fastspeech/Screen_Shot_2022-06-21_at_4.37.52_PM.png)

실제 목소리(GT)와 비교했을 때 tacotron2나 transformer TTS와 함께 굉장히 높은 점수가 나오는 것을 확인할 수 있다. 이는 사람들이 들었을 때 실제 사람과 거의 유사하다고 볼 수 있다.

## 3-2. Inference Speedup

![Screen Shot 2022-06-21 at 4.40.19 PM.png](/assets/posts/TTS/fastspeech/Screen_Shot_2022-06-21_at_4.40.19_PM.png)

FastSpeech의 강점은 inference speed에서 나타난다. Autoregressive 방식을 사용하는 transformer TTS와 비교했을 때 mel spectrogram 생성을 비교하면 약 270배 빠른 속도로 결과가 나오고 neural vocoder을 사용해 오디오까지 생성했을 때 38배 빠른 속도로 변환이 진행되었다.

![Screen Shot 2022-06-21 at 4.43.26 PM.png](/assets/posts/TTS/fastspeech/Screen_Shot_2022-06-21_at_4.43.26_PM.png)

이전 TTS 모델들의 치명적인 단점 중 하나는 입력 문장의 길이가 길어지면 자연스러움이나 억양이 크게 떨어지는 것이다. Transformer을 사용하는 transformer TTS와 FastSpeech는 이 부분에서 큰 robustness를 보여준다. 문장 길이에 따라 기하급수적으로 늘어나는 이전 모델들의 inference time과 비교했을 때 선형적으로 증가하지만, autoregressive 방식을 사용하는 transformer TTS와 비교했을 때 굉장히 짧은 시간 안에 연산이 끝나는 것을 확인할 수 있다.

## 3-3. Robustness

![Screen Shot 2022-06-21 at 4.46.41 PM.png](/assets/posts/TTS/fastspeech/Screen_Shot_2022-06-21_at_4.46.41_PM.png)

Transformer와 Length Regulator을 사용함으로 인해 위 결과와 같이 말을 반복하거나 스킵하는 경우, 발음이 부정확하거나 다른 경우가 0에 수렴하여 Error Rate에서 획기적인 개선을 한 것을 알 수 있다.

## 3-4. Length Control

![Screen Shot 2022-06-21 at 7.10.09 PM.png](/assets/posts/TTS/fastspeech/Screen_Shot_2022-06-21_at_7.10.09_PM.png)

2-2에서 설명했던 Length Regulator에서 alpha 값을 조정했을 때 같은 문장이라도 말하는 속도를 조절할 수 있는 기능에 대한 결과값이다. 스펙트로그램 hidden state size를 조정함으로써 임의로 오디오를 빠르거나 느리게 감는 것이 아닌 실제 빠르고 느리게 말하는 자연스러움을 얻을 수 있다.

![Screen Shot 2022-06-21 at 7.16.29 PM.png](/assets/posts/TTS/fastspeech/Screen_Shot_2022-06-21_at_7.16.29_PM.png)

말을 할 때 중간에 공백을 주는 것이 prosody에 큰 영향을 끼치는데, FastSpeech에서는 입력 문장에 스페이스 글자의 수를 조정함으로써 이와 같은 변화를 줄 수 있다. 위 오른쪽 스펙트로그램은 왼쪽의 입력 텍스트에 스페이스를 여러번 추가해 단어와 단어 사이에 공백을 추가한 것인데, 완벽한 빈칸은 아니지만 에너지가 거의 존재하지 않는 구간(소리가 거의 들리지 않는 구간)이 잘 생성되었고, 왼쪽보다 길이가 늘어난 것을 확인할 수 있다.

## 3-5. Ablation Study

![Screen Shot 2022-06-21 at 7.19.11 PM.png](/assets/posts/TTS/fastspeech/Screen_Shot_2022-06-21_at_7.19.11_PM.png)

Ablation study에서는 CMOS metric을 사용해 출력 오디오의 퀄리티를 baseline 모델과 비교한다. FFT 모듈에서 1D convolution과 teacher model의 contribution을 목적으로 진행했고 모두 학습에 의미 있는 기여를 한다는 것을 확인했다.

# 4. Conclusions

FastSpeech는 이전 모델들보다 획기적인 빠른 속도와 낮은 에러율, 높은 자연스러움을 선보이는 TTS 모델이다. Feed-Forward Transformer, Length Regulator, Duration Predictor의 사용을 통해 가능했다. 목소리 생성 뿐만 아니라 말하는 속도 조절과 공백 생성도 쉽게 가능한 모델이다.

# 5. Future Work

- 단일화자 데이터셋이 아닌 다화자 데이터셋으로 학습 + low resource로 학습을 진행할 예정이다.
- 스펙트로그램에서 오디오로 변환하는 neural vocoder을 동시에 학습시키는 방향을 시도할 예정이다.
