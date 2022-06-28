---
layout: post
title:  "[Paper Review] FastSpeech2: Fast and High-Quality End-to-End Text-to-Speech"
author: 김재훈
categories: [Deep Learning, Text-to-Speech, Speech Synthesis, Transformer]
image: assets/images/logo-speech.jpeg
---

# 1. Overview

FastSpeech2 모델은 이전 모델인 FastSpeech 모델의 단점을 보안하고, text로부터 스펙트로그램이 아닌 waveform을 생성하는 end-to-end 모델인 FastSpeech2s를 제시한다. FastSpeech2는 3배 단축된 학습 시간이 소요되고 pitch, energy 정보를 조정함에 따라 더 자연스러운 발화를 생성할 수 있다.

# 2. Methods

논문에서 제안한 FastSpeech의 단점은 다음과 같다.

- teacher-student distillation 과정이 복잡하고 많은 시간이 걸린다.
- teacher 모델에서 추출한 duration information이 정확하지 않고, 추출한 스펙트로그램은 학습 target으로 사용하기에 information loss가 심하다.

FastSpeech2 모델은 위 단점을 보완하기 위해 다음과 같은 기술을 사용했다.

- teacher 모델에서 추출한 스펙트로그램이 아닌 ground-truth audio의 스펙트로그램을 사용해 학습한다.
- acoustic model이 잘 학습될 수 있도록 pitch, energy, duration 정보를 예측하는 모듈을 추가해 학습한다.

![Untitled](/assets/posts/TTS/fastspeech2/Untitled.png)

## 2-1. Variance Adaptor

FastSpeech2에서 새롭게 등장한 기능들 중 하나는 Variance Adaptor이다. FastSpeech는 phoneme마다 해당하는 스펙트로그램의 개수를 예측하는 duration predictor에서 발전해 duration/pitch/energy predictor이 탑재된 variance adaptor은 텍스트에 해당하는 발화의 acoustic information을 더 잘 학습할 수 있도록 도와준다. 

### Duration Predictor

해당 모듈은 FastSpeech와 역할은 비슷하지만, 더 정확한 정보를 통해 개선되었다. 이전 학습 방식은 teacher 모델에서 monotonic attention을 추출한 후 duration sequence를 계산해 phoneme-spectrogram dimension matching을 진행했다. 

하지만 추출한 attention이 duration mapping을 정확하게 가지고 있지 않을 경우가 있고, 보다정확한 정보를 통해 학습하자는 것이 motivation이다.

FastSpeech2는 해당 모듈을 trainable하게 만들고 Montreal Forced Alignment (MFA)라는 phoneme duration을 추출하는 프로그램을 통해 ground truth를 생성하고 ground truth와 비교를 통해 alignment를 학습할 수 있도록 만들었다. duration 정보의 학습은 MSE loss를 기반으로 이루어진다.

### Pitch Predictor

높은 퀄리티의 출력값을 위해 pitch 정보를 사용하는 다른 TTS 모델들과 같이 pitch predictor을 사용했다. 하지만 pitch는 높은 variation을 가지기 때문에 그대로의 값을 사용하지 않고 ground truth pitch spectrogram을 생성하고 이를 예측하도록 한다.

Ground truth 오디오 데이터로부터 continuous pitch series를 계산하고, 이를  Continous Wavelet Transform (CWT) 변환을 사용해 pitch spectrogram으로 만든다.  Pitch predictor은 해당 스펙트로그램을 타겟으로 MSE loss를 통해 학습한다.

Inference 상황에서 pitch predictor에서 예측된 pitch spectrogram은 inverse CWT를 통해 pitch series로 변환된 후 목소리 생성에 도움을 주게 된다. Pitch contour을 사용할 때 편의를 위해 log-scale 256개로 quantize한 후 pitch embedding으로 변환해 text 정보를 가지고 있는 hidden sequence에 더한다.

### Energy Predictor

스펙트로그램에서 에너지는 소리의 크기 뿐만 아니라 fundamental frequency와 prosody에도 중요한 영향을 미친다. Energy predictor은 스펙트로그램에서 각 time frame마다 L2-norm 에너지를 계산, log-scale 256개로 quantize한 후 ground truth의 에너지와 비교해 학습한다 (loss function은 MSE Loss를 사용한다).

## 2-2. FastSpeech2s

FastSpeech와 FastSpeech2는 모두 Text를 입력으로 받아 (mel) spectrogram을 출력하는 모델(acoustic model)이다. 스펙트로그램에서 오디오로 변환하는 과정은 neural vocoder의 힘을 빌렸는데, 논문에서는 FastSpeech2에서 spectrogram decoder이 아닌 audio decoder 모듈을 만들고 이를 사용한 fully end-to-end 모델을 FastSpeech2s로 지정했다.

### Challenges

텍스트를 입력으로 받아 오디오를 출력하는 end-to-end 모델이 적은 이유는 크게 두가지가 있다.

1. 스펙트로그램과 비교했을 때 오디오 데이터는 더 많은 variance information을 가지고 있다 (magnitude, phase, etc.)
2. 오디오 데이터의 크기가 너무 커 학습이 어렵다
    1. 기본적인 오디오 데이터는 15kHz 샘플링으로 이루어져 있는데, 1초에 16,000개의 데이터포인트가 있다
    → 멜 스펙트로그램과 비교했을 때 80 X 200 = 16,000 데이터는 6초 가량의 정보를 가지는 것과 비교하면 긴 오디오 데이터를 학습하려면 메모리 크기가 아주 커야 한다.

### Decoder Architecture

waveform decoder은 위 언급한 어려움을 극복하기 위해 adversarial training을 진행해 phase information을 예측할 수 있도록 학습한다. Discriminator은 높은 퀄리티의 waveform 생성으로 유명한 Wavenet의 설계를 기반으로 dilated non-causal 1D convolution과 leaky ReLU를 사용해 제작되었다.

hidden state과 waveform의 dimension을 맞추기 위해  transposed 1D convolution을 사용하고, 비교적 멀리 있는 정보들을 기반으로 퀄리티를 개선하기 위해 dilated convolution을 사용했다. 

**FastSpeech2s는 FastSpeech2를 제시함과 동시에 fully end-to-end가 가능하도록 만든 것이 주 목적으로 보이고, neural vocoder을 사용한 결과들과 비교했을 때 경쟁력은 떨어지는 것으로 보인다.**

# 3. Results

## 3-1. Inference MOS

![Screen Shot 2022-06-28 at 7.19.12 PM.png](/assets/posts/TTS/fastspeech2/Screen_Shot_2022-06-28_at_7.19.12_PM.png)

FastSpeech와 마찬가지로 FastSpeech2와 FastSpeech2s의 출력값에 대한 정성적인 지표인 Mean Opinion Score (MOS)를 비교한다. FastSpeech는 이전 모델들과 비교했을 때 non-autoregressive 방식을 사용해 inference speedup이 가장 큰 contribution이었기 때문에 tacotron2, transformer TTS와 비교했을 때 MOS 점수가 낮지만, pitch, energy와 같은 추가적인 acoustic information을 통해 성능을 개선한 FastSpeech2는 이전 모델보다 압도적인 MOS 점수를 보인다. **FastSpeech2는 inference 속도가 훨씬 빠를 뿐만 아니라 출력 오디오 퀄리티도 느린 autoregressive TTS 모델들보다 뛰어나게 되었다**. 앞서 언급했던 것과 같이 FastSpeech2s 모델은 end-to-end 파이프라인을 형성하는 것이 주된 목적이었기 때문에 MOS는 비교적 낮은 것을 볼 수 있다.

![Screen Shot 2022-06-28 at 7.19.56 PM.png](/assets/posts/TTS/fastspeech2/Screen_Shot_2022-06-28_at_7.19.56_PM.png)

FastSpeech2는 간소화된 파이프라인을 통해 경량화되었고, inference speedup 뿐만 아니라 training time 또한 크게 줄어든 것을 알 수 있다. FastSpeech2s의 경우 waveform decoder을 하는 adversarial training 과정이 시간을 많이 소요해 이전 모델들보다 긴 학습 시간이 기록된 것으로 추측된다.

## 3-2. Analysis on Variance Information

![Screen Shot 2022-06-28 at 7.26.10 PM.png](/assets/posts/TTS/fastspeech2/Screen_Shot_2022-06-28_at_7.26.10_PM.png)

위 테이블에서 sigma, gamma, kappa 변수들은 각각 deviation, skewness, kurtosis를 나타내며, waveform data distribution의 nth moment를 표현한다. Ground truth의 moment와 값이 유사할 수록 invariant하다고 할 수 있다.

Dynamic Time Warping (DTW) distance는 오디오 데이터 사이의 유사도를 나타내는 distance cost이다.

위 테이블에서 알 수 있듯이 FastSpeech2와 FastSpeech2s의 moment 값들이 ground truth와 가장 유사한 것을 알 수 있다. 특히 FastSpeech와 비교했을 때 kurtosis 값이 크게 개선되어 ground truth 값과 유사해졌다.

## 3-3. Ablation Study

![Screen Shot 2022-06-28 at 7.33.59 PM.png](/assets/posts/TTS/fastspeech2/Screen_Shot_2022-06-28_at_7.33.59_PM.png)

FastsSpeech와 비교했을 때 오디오 퀄리티 개선에 가장 큰 의미가 있는 pitch, energy prediction 모듈에 대한 ablation study를 진행했다. Comparative Mean Opinion Score (CMOS)는 MOS 값에 대한 비교적인 지표로, 대부분ablation study에서 같은 텍스트로 만들어진 오디오 데이터에 대한 평가로 사용된다. CMOS가 음수 값을 가지면 기준 오디오보다 퀄리티가 떨어지는 것으로 해석할 수 있다.

energy도 전반적인 스펙트로그램 생성에 있어 중요하지만, FastSpeech2에서는 pitch prediction을 뺏을 때 energy prediction을 뺀 것보다 CMOS가 크게 떨어지는 것을 확인할 수 있다.

# 4. Conclusion

논문에서 제시한 FastSpeech2는 pitch, energy information을 추가적으로 학습함으로써 이전 FastSpeech보다 더 나은 성능을 보여준다. Teacher-student knowledge distillation 없이 external alignment label을 사용해 training pipeline을 간소화함과 동시에 모델 경량화를 통해 학습과 inference 시간을 대폭 단축시켰다.

FastSpeech2 설계를 기반으로 스펙트로그램이 아닌 raw waveform을 출력하는 fully end-to-end TTS 모델을 제시한다. waveform decoder을 통해 짧은 오디오 데이터를 텍스트로부터 바로 출력할 수 있다.

# 5. Future Work

- 해당 모델은 학습 시 MFA를 통해 alignment 정보를 추출하는 전처리 과정을 거친다. 저자는 위 과정 또한 텍스트나 오디오 정보로부터 native하게 뽑아내고, 따라서 해당 로직을 발견 및 적용한다면 파이프라인 간소화와 모델 경량화를 추가적으로 진행할 수 있을 것 으로 본다.
- FastSpeech2s는 fully end-to-end 시스템이지만 앞서 언급한 raw audio waveform을 사용하는 점에 대해 memery limiation이 존재한다. 논문에서 제시한 모델은 아주 짧은 길이의 오디오만 end-to-end로 출력할 수 있지만 이후 추가적인 방법을 통해 긴 오디오를 end-to-end로 뽑아내는 것을 목표로 한다.