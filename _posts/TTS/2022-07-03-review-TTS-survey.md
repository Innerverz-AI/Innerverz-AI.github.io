---
layout: post
title:  "[Paper Review] A Survey on Neural Speech Synthesis"
author: 김재훈
categories: [Deep Learning, Text-to-Speech, Speech Synthesis]
image: assets/images/logo-speech.jpeg
---

# 1. Overview

본 리뷰에서는 TTS 모델을 이해하기 위한 기본적인 기술들과 용어들에 대한 설명과 현재 높은 성능을 가지는 잘 알려져 있는 모델들이 사용하는 알고리즘에 대한 소개로 이루어져 있다. TTS라는 주제는 오랜 기간 연구된 분야이기 때문에 그 안에서 일어나는 모든 일들을 설명할 수는 없지만, 전반적인 파이프라인과 기본적인 딥러닝, 오디오 신호처리의 개념을 알고 있다면 엔지니어링의 관점에서 학습이나 inference에 도움이 될 것으로 보인다.

# 2. History of TTS technology

인공지능을 활용하지 않은 과거의 TTS 기술들은 크게 네가지로 나뉘며, 각자 장단점을 가진다. 딥러닝을 사용한 TTS 모델은 2010년 이후 크게 발전했으며, 마지막으로 설명하는 Statistical Parametric Synthesis의 일종이다.

## 2-1. Articulatory Synthesis

해당 방식은 말하는 소리를 생성하는 가장 근본적인 방법으로, articulation(정확하게 말하는 것)을 위해 성대, 입술, 혀 등을 구현해 실제 사람이 말하는 방식과 유사한 상황을 만드는 것을 목표로 한다.

하지만 해당 방식은 하드웨어적인 면으로 봤을 때 구현이 상당히 까다롭고 소프트웨어적으로 봤을 때 혀의 위치나 입술의 모양과 같은 데이터는 굉장히 모으기 어렵기 때문에 이를 기반으로 data-driven, rule-based 모델링을 하기에 쉽지 않아 현존하는 모델들은 다른 방식들에 비해 퀄리티가 현저하게 떨어진다.

## 2-2. Formant Synthesis

Formant는 사람의 목소리를 인지할 때 화자의 캐릭터와 목소리 퀄리티를 결정하는 중요한 요소들 중 하나이다. 이는 시간의 변화에 따른 주파수 스펙트럼을 표현하는 spectrogram에서 찾아볼 수 있다.

![Untitled](/assets/posts/TTS/TTS-survey/Untitled.png)

위 스펙트로그램에서 에너지가 높은 구간을 연결한 다양한 색의 선들 중 가장 아래에 있는 노란색 선은 목소리의 높낮이(pitch)를 (대부분) 결정하는 fundamental frequency(F0)이다. 그 위에 있는 선들이 formant인데, 아래부터 순서대로 F1, F2, F3로 분류된다. Formant에 대한 연산과 이를 통한 발화 생성은 단편적으로 봤을 때 간단한 rule-based 연산으로 발화를 만들 수 있기 때문에 embedded system에서 주로 사용되어 왔다.

딥러닝을 활용한 TTS 모델링 이전 신호처리를 통한 음성합성에서 formant는 굉장히 중요한 역할을 했고, 단어나 음소(phoneme)별로 formant가 어떻게 생성되는지에 대한 정보를 기반으로 스펙트로그램을 생성했으며, 딥러닝을 활용한 TTS 모델들도 formant information을 사용하는 모델들이 자연스러운 음성을 생성하는 것으로 알려져 formant가 음성합성에 중요한 정보라는 것이 다시 한번 강조된다. 하지만 굉장히 nonlinear한 목소리의 특성 때문에 formant distribution에 대한 특정 패턴을 규정하기 어려워 아직까지 연구가 이루어지고 있다.

## 2-3. Concatenative Synthesis

해당 방식은 녹음된 여러 단어나 글자의 조합을 통해 문장을 생성한다. 쉽게 말해 ARS를 사용할 때 ‘0’, ‘1’, ‘0’, …과 같이 숫자 하나하나를 녹음해 두고 전화번호에 해당하는 녹음 파일들을 합치는 방식이다. 

해당 방식은 간단한 음성합성 모델(숫자 말하기)들의 경우 낮은 비용으로 높은 인지력을 가질 수 있지만, 녹음되지 않은 발음은 생성할 수 없거나 글자와 글자 사이가 자연스럽게 이어지지 않는 등 퀄리티적인 면에서 개선이 굉장히 어렵고 글자마다 녹음해야 하기 때문에 굉장히 큰 데이터베이스를 필요로 한다.

## 2-4. Statistical Parametric Speech Synthesis (SPSS)

해당 방식은 현재 많이 발전된 딥러닝 TTS 모델들의 기반이 되는 방식으로, 단순한 오디오들을 concatenate하는 것이 아니라 acoustic parameter에 대한 분석을 기반으로 글자나 단어 사이의 관계를 통해 자연스럽게 이어지는 목소리를 생성하는 것을 목표로 한다.

해당 방식은 보편적으로 텍스트 분석 →  acoustic parameter 예측 → vocoder을 통한 오디오 생성의 3단계를 가진다.

![Screen Shot 2022-07-04 at 2.42.27 AM.png](/assets/posts/TTS/TTS-survey/Screen_Shot_2022-07-04_at_2.42.27_AM.png)

### 텍스트 분석

텍스트 분석은 text normalization, grapheme-to-phoneme conversion, word segmentation과 같이 텍스트 데이터와 발화 음성간의 관계를 가깝게 해주는 과정이다. 예를 들어 주어진 텍스트가 “1998년"이라면 해당 텍스트를 발화했을 때의 발음인 “천구백구십팔년"으로 변환하는 text normalization을 통해 이후 단계에서 목표하는 발화 오디오에 필요한 정보를 더 쉽게 알 수 있도록 한다.

### acoustic parameter 예측

발화 정보를 잘 포함하는 텍스트를 통해 말의 높낮이, 발음과 발음 사이의 전환과 같은 부분들을 책임지는 acoustic parameter에 대한 예측이 진행된다. 딥러닝 이전의 TTS 모델들은 텍스트가 가지는 linguistic feature을 Hidden Markov Model(HMM)을 기반한 확률을 통해 스펙트로그램이나 cepstrum (발화에서 source를 제외한 filter(발음) 정보만을 가지는 데이터) 정보를 예측했다.

최근 높은 성능을 보이는 TTS 모델들은 DNN을 활용해 HMM보다 acoustic parameter을 잘 예측하게 되고, 많은 데이터를 기반으로 robustness가 개선된 버전이라고 할 수 있다.

### vocoder을 통한 오디오 생성

vocoder은 이전 과정들을 통해 얻어진 발화에 대한 정보를 기반으로 오디오 waveform을 생성한다. 스펙트로그램이나 cepstrum을 기반으로 Griffin-Lim과 같은 알고리즘을 사용해 frequency / cepstrum distribution → audio waveform 변환 과정을 가지는데, 비교적 independent한 정보의 합으로 생성되는 다른 접근들과는 달리 확률적 모델링을 통한 acoustic information으로 생성된 SPSS의 오디오는 자연스러움과 다양한 발화에 대해 유연하게 바뀔 수 있는 장점을 가진다.

하지만 acoustic information과 waveform 사이에 존재하는 information gap (magnitude-phase relationship)을 예측하는 과정에서 불필요한 artifact가 발생하고 이는 생성되는 오디오의 퀄리티를 현저하게 저하시킨다. 최근 딥러닝을 통해 Griffin-Lim이 아닌 Generative model을 통해 스펙트로그램으로 오디오를 생성하는 neural vocoder이 좋은 성능을 내고 있다.

## 2-5. Neural Speech Synthesis

이전에 언급했던 바와 같이 Neural Network를 사용한 TTS 모델들은 많은 데이터와 빠른 연산으로 음성합성에서 속도와 퀄리티를 모두 개선시켰다.

WaveNet은 최초로 발표된 linguistic feature을 통한 waveform 생성 모델이다. TTS만을 위해 만들어진 것은 아니지만 현재까지도 해당 모델에서 사용한 dilated non-causal convolution과 같은 알고리즘은 오디오 인공지능 분야에 다양하게 사용되고 있다.

DeepVoice 1, 2 모델의 경우 앞서 설명했던 acoustic parameter 예측 모듈을 neural network를 기반으로 설계해 성능을 크게 개선했다.

Phoneme(음소)는 텍스트에서 추출 가능한 발음의 기본 단위로, 텍스트로부터 acoustic information을 추출하는데 중요한 역할을 한다. 이 때문에 시작부터 phoneme을 입력으로 받는 TTS 모델 분야에서 많은 발전이 이루어졌고, 가장 잘 알려져 있는 tacotron 1, 2, DeepVoice 3, FastSpeech 1, 2 모델은 텍스트 분석 모듈을 간소화하고 NN 모델이 phoneme sequence를 입력으로 받아 발화를 출력하도록 하였다. 해당 모델들은 waveform이 아닌 spectrogram을 출력하도록 설계되었고, 이후 neural vocoder이나 Griffin-Lim 알고리즘을 사용해 waveform을 생성한다.

FastSpeech 2s, ClariNet, EATS 모델들은 vocoder 모듈까지 하나의 모델에 결합시켜 텍스트를 입력으로 받았을 때 waveform을 출력할 수 있는 fully end-to-end TTS 모델이다.

# 3. Specific Topics in TTS

![Screen Shot 2022-07-04 at 2.43.19 AM.png](/assets/posts/TTS/TTS-survey/Screen_Shot_2022-07-04_at_2.43.19_AM.png)

Neural TTS 모델들의 전반적인 taxonomy는 위 분류와 같이 여러개로 나누어진다. 이후 알고리즘 설명에서 해당되는 다양한 모델들을 참조하겠지만, 언급하지 않은 모델들의 경우 큰 분류에서 비슷한 알고리즘으로 진행되는 것으로 이해하면 좋을 것 같다.

해당 survey는 유명한 TTS 모델들 중 하나인 FastSpeech 1, 2 모델을 발표한 microsoft에서 연구진이 발표한 논문으로, 다양한 관점에서 현재 나와있는 모델들을 비교 분석하였다. 이번 리뷰에서는 이 중 중요한 역할을 하는 acoustic model과 vocoder에 대한 설명을 다룬다.

![Screen Shot 2022-07-04 at 2.43.39 AM.png](/assets/posts/TTS/TTS-survey/Screen_Shot_2022-07-04_at_2.43.39_AM.png)

위 dataflow는 발표된 모델들이 TTS 파이프라인에서 어느 부분을 담당하는지 알기 쉽게 설명한다. 앞서 언급한 대로 phoneme의 중요성 때문에 다양한 모델들이 text가 아닌 phoneme 정보에서 시작해 발화를 출력하는 것을 볼 수 있고, character에서 waveform으로 바로 연결되는 end-to-end 모델들도 내부적으로 grapheme(문자, 문자소)-to-phoneme(g2p) 모듈이 내장되어 있는 경우가 많다 (FastSpeech 2s, Wave-Tacotron, etc.)

## 3-1. Acoustic models

TTS의 출력값인 발화 오디오 데이터는 continuous waveform으로, 사진, 영상, 비디오와 비교했을 때 temporal dependency가 훨씬 크다. 따라서 adjacent information을 잘 활용할 수 있는 CNN, time-varying information을 잘 활용하는 RNN과 attention mechanism, 과거의 정보를 기반으로 미래를 예측하는 causal autoregressive mechanism, 그리고 RNN + attention을 병렬화해 속도를 크게 개선한 transformer이 주로 사용된다. 

### RNN-based Models (tacotron series)

NVIDIA에서 발표한 딥러닝을 사용한 거의 최초의 모델은 tacotron은 encoder-attention-decoder 파이프라인을 통해 acoustic information을 예측한다. RNN + attention을 사용해 텍스트로부터 추출한 linguistic information에서 acoustic information을 예측하고, linear 스펙트로그램을 출력 후 Griffin-Lim 알고리즘을 통해 waveform 생성한다.

> Griffin-Lim 알고리즘은 linear 스펙트로그램에서 waveform을 변환할 때 사용되는데, waveform과 스펙트로그램 사이의 변환을 책임지는 Short-Time Fourier Transform(STFT)은 complex domain에서 이루어지지만 임의로 생성되는 스펙트로그램의 경우 real value를 가지기 때문에 magnitude 정보만 가지고 있다. 이 경우 inverse STFT가 불가능하기 때문에 Griffin-Lim 알고리즘은 iterative method를 통해 phase information을 예측하고, 결과적으로 waveform을 생성하게 된다. 하지만 예측을 통해 얻어진 phase 정보가 완벽하지 않기 때문에 해당 알고리즘으로 생성된 오디오는 특유의artifact가 들리게 되고, 오디오 퀄리티를 현저히 저하시킨다.
> 

tacotron 2는 tacotron에서 acoustic model을 개선하고 linear spectrogram이 아닌 mel spectrogram을 생성한 후 WaveNet과 유사한 vocoder을 사용해 발화를 출력하면서 성능을 대폭 상승시켰다.

RNN의 사용으로 시간에 따라 변화하는 정보를 잘 예측하고 attention mechanism으로 중요한 정보에 대한 mapping을 통해 당시 높은 퀄리티의 발화를 추출해 유명해졌으나, 이전 연산을 기다려야 하는 RNN이 가지는 특유의 느린 속도와 Griffin-Lim을 통한 waveform reconstruction에서 artifact가 생기는 것이 단점이다.

### CNN-based Models (DeepVoice series, WaveNet)

오디오 도메인에서 CNN은 waveform과 discrete information과의 거리를 좁혀주는데에 가장 많이 사용되며, DeepVoice 1, 2와 WaveNet에서도 acoustic information에서 waveform을 생성하는데 사용되었다. 

> 해당 분야는 vocoder과 경계가 애매한데, 개인적인 생각으로는 generative model을 사용해 discriminator을 다양한 방식으로 사용하는 부분에서 vocoder 분야로 칭해지고 CNN을 사용한 기본적인 waveform 생성은 포함되지 않는 것 같다.
> 

이전에 언급했던 대로 WaveNet은 처음으로 waveform을 생성하는 인공지능 모델로, dilated + causal convolution을 사용했다. kernel dilation을 사용해 기준점으로부터 보다 멀리 있는 부분에서까지 정보를 받아올 수 있게 된다 (이는 receptive field를 증가시킨다고 표현한다). causality를 활용함으로써 autoregressive 알고리즘을 사용하게 되는데, 이는 과거의 정보만을 통해 바로 다음의 미래를 예측하는 것이다. 스펙트로그램을 타겟으로 했을 때, 첫 프레임을 생성하는 데에는 해당 프레임에 대한 acoustic information만을 가지고 예측하지만, 이후 N번째 스펙트로그램 프레임을 예측할 때는 첫번째부터 N-1번째 프레임의 정보를 가지고 예측하기 때문에 점점 성능이 좋아지고 프레임들이 자연스럽게 이어진다.

DeepVoice는 WaveNet과 거의 유사한 설계를 가지고, DeepVoice 2는 DeepVoice 설계를 조금 변경해 multi-speaker 발화를 출력할 수 있도록 제작되었다. DeepVoice 3는 synthesis 과정에서 fully convolutional computation을 사용하고 tacotron 2와 유사하게 mel spectrogram 생성 후 WaveNet을 통해 waveform을 생성한다.

autoregressive 알고리즘은 높은 퀄리티의 성능을 보인다는 장점이 있지만, N번째 예측을 위해서는 이전 프레임의 생성이 완료되어야 하기 때문에 병렬연산이 거의 불가능하고, 속도적인 면에서 아주 나쁘다고 볼 수 있다.

### Transformer-based Models (FastSpeech series)

Transformer은 RNN을 사용하지 않고 self-attention mechanism으로 temporal dependency와 병렬 연산을 모두 가능하게 한 것으로 유명하다. tacotron의 경우 attention을 사용하지만 학습이 잘 되지 않았을 경우 attention이 monotonic alignment를 가지게 않아 같은 말을 반복하거나 특정 텍스트를 발화하지 않고 건너뛰는 경우가 발생한다. DeepVoice, WaveNet과 transformer을 사용하는 transformerTTS 모델들은 autoregressive 방식을 사용하기 때문에 높은 성능을 가지지만 긴 발화의 경우 많은 프레임들을 예측하는 과정에서 상당한 시간이 소요된다.

위 문제점들을 microsoft에서 발표한 FastSpeech 모델은 encoder-decoder 설계를 가지는 transformer이 아닌 Feed-Forward Transformer(FFT)를 사용해 해결했다. Multi-head self-attention으로 alignment를 잘 학습하고 내부에서 1D convolution을 통해 adjacent information dependency를 효과적으로 학습할 수 있도록 만들었다. text(phoneme) sequence와 스펙트로그램 사이의 dimension은 항상 다른데, 보통 한 phoneme을 발화하기 위해서는 둘 이상의 스펙트로그램 프레임을 필요로 한다. 이 one-to-many mapping 문제를 duration prediction 모듈로 해결하고, 학습된 모듈을 통해 acoustic information을 예측하고 결과적으로 빠른 속도로 자연스러운 발화를 합성할 수 있게 된다.

FastSpeech 2 모델은 FastSpeech에서 중요한 역할을 하는 duration prediction 모듈을 외부 alignment 정보로 대체하고, pitch, energy와 같은 acoustic information을 직접 예측하는 모듈을 supervised learning으로 학습시켜 출력 발화 퀄리티를 더욱 개선하였다.

### Model Summary

앞서 설명한 모델들과 다른 발표된 TTS acoustic model들을 분류한 테이블을 아래에서 확인할 수 있다. 

![Screen Shot 2022-07-04 at 3.07.41 AM.png](/assets/posts/TTS/TTS-survey/Screen_Shot_2022-07-04_at_3.07.41_AM.png)

## 3-2. Vocoders

vocoder은 linguistic feature에서 바로 waveform을 추출하는 low-level vocoder도 존재하지만, 대부분 스펙트로그램(linear, mel)에서 waveform 변환을 도와주어 Griffin-Lim 알고리즘보다 높은 퀄리티의 발화를 생성하는 모델이 다양하게 개발되었다. 

### Autoregressive Vocoders

Griffin-Lim은 phase 정보를 직접적으로 예측해 waveform을 생성하지만, NN-based vocoder은 해당 guidance가 없기 때문에 학습과 inference에서 추가적인 도움이 필요하다. 시간과 같은 방향으로 autoregressive한 모델이 따라서 높은 성능을 보여주었고, WaveNet을 포함해 WaveRNN, Char2Wav와 같은 모델들이 neural vocoder 분야의 시작을 성공적으로 이끌었다.

이후 autoregressive의 느린 단점을 보완하기 위해 다양한 모델이 발표되었고, GAN-based non-autoregressive 모델들이 압도적인 성능을 보여 주목받았다.

### GAN-based Vocoders

Generative Adversarial Network(GAN)은 생성 task에 높은 성능을 보이는 것으로 유명해 오디오 도메인에서 vocoder 분야에서 많이 사용된다.

Vocoder에서 generator은 대부분 dilated convolution과 transposed convolution을 사용한다. 앞서 설명한 다른 모델들과 같은 이유로 dilated convolution을 사용해 long-term dependency를 잘 학습할 수 있도록 하고, 스펙트로그램과 waveform의 dimension discrepancy를 극복하기 위해 transposed convolution을 사용해 upsampling을 진행한다. 높은 성능으로 유명한 MelGAN, VocGAN, HiFi-GAN 모델들이 모두 해당 레이어를 사용하고, hidden state에서 갑자기 프레임이 늘어나지 않도록 multi-scale upsampling을 진행한다.

> 스펙트로그램은 보편적으로 초당 300프레임을 가지지만, waveform의 경우 초당 16000개 이상의 데이터포인트를 가진다. (약 512배)
> 

Discriminator의 역할 또한 중요하다. GAN-TTS의 경우 random window discriminator을 사용해 출력값에서 랜덤한 부분을 선정해 판별한다. MelGAN의 경우 multi-scale discriminator을 사용해 출력된 오디오를 다양하게 downsampling한 후 판별을 진행한다. VocGAN은 hierarchical discriminator을 사용해 낮은 주파수부터 높은 주파수까지 acoustic information mapping이 잘 이루어지는지 판별한다. Hifi-GAN의 경우 출력된 오디오를 각각 다른 주기로 나눈 2D 데이터로 변환한 후 판별한다. 현재 가장 높은 성능을 보이는 모델은 Hifi-GAN이지만, 학습 시간이 다른 모델들에 비해 훨씬 오래 걸리는 편이다.

### Summary

앞서 설명한 vocoder의 분류가 아래 표에 정리되어 있다. 분류가 시간 순서는 아니지만, AR을 사용하는 vocoder은 비교적 오래 전에 발표되었고 이에 대한 단점을 보완한 NAR + GAN을 사용하는 vocoder은 최근에 발표된 경향이 있다. 최근에 발표된 높은 성능의 vocoder들은 모두 CNN을 사용하는 것 또한 확인할 수 있다.

![Screen Shot 2022-07-04 at 4.05.55 AM.png](/assets/posts/TTS/TTS-survey/Screen_Shot_2022-07-04_at_4.05.55_AM.png)

dataflow별로 다르게 적용되는 vocoder은 아래 그림을 통해 쉽게 확인할 수 있다.

![Screen Shot 2022-07-04 at 4.37.48 AM.png](/assets/posts/TTS/TTS-survey/Screen_Shot_2022-07-04_at_4.37.48_AM.png)

# 4. Summary + Future Works

현재까지 발표된 TTS 모델들의 관계도는 다음과 같다. 가장 보편적으로 사용되는 모델은 FastSpeech1, 2 모델이며, 가장 최근에 발표된 AdaSpeech 3 모델은 Voice Cloning task의 일종으로 발화 생성에 있어 입력으로 텍스트와 함께 타겟 화자의 목소리가 들어가게 되고, 출력 발화가 타겟 화자와 유사한 질감을 가지는 것을 목표로 한다.

![Screen Shot 2022-07-04 at 4.38.50 AM.png](/assets/posts/TTS/TTS-survey/Screen_Shot_2022-07-04_at_4.38.50_AM.png)

현재 neural network를 활용한 TTS 모델들은 꽤나 좋은 성능과 빠른 속도를 가지지만, 다음과 같은 개선점들을 여전히 가지고 있다.

- 적은 labeled data를 사용한 low resource training
- 화자 정보나 감정 요소가 포함된 multi-speaker / expressive speech synthesis
- AdaSpeech 3와 같이 타겟 화자 목소리를 짧게 듣고 이를 다양하게 따라할 수 있는 Voice Cloning

해당 목표들은 꾸준히 연구되고 있고, 딥러닝 모델의 개선 뿐만 아니라 신호처리나 인지공학적인 domain knowledge를 요구하는 접근이 다양하게 연구되고 있다.