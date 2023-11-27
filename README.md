# Audio-Generation

Our extensive exploration voice generation has led to the implementation of promising approaches to improve the one. Despite challenges in achieving substantial improvements, the deployment of a neural network for voice slicing and an overarching denoising model has shown potential for enhancing the training process and refining the audio output.


## Data

The model is trained on publicly available audio packages, using AVSPEECH to obtain pure speech signals. Out of 2448 videos, we selected 2214 for training and 234 for testing.

We use two datasets as background noise: DEMAND and AudioSet from Google.

Using the linearity of sound wave propagation, we synthesize noisy input signals by choosing a signal-to-noise ratio of seven values. For example, with an SNR of -10 dB, the noise power is ten times the power of speech. The SNR range in our estimates (-10 dB, 10 dB) is much wider than in previous studies.

## Preprocessing

Our primary objective in adopting this approach is to elevate the precision and reliability of the speech fragment extraction process, thereby bolstering the overall efficacy of the training process. Through this analytical framework, we seek to harness the amalgamation of information from randomized slices to cultivate an innovative and enhanced audio output.

On a complementary note, we have incorporated an overarching denoising model (refer to Figure 1) designed to eliminate extraneous noise from the entirety of the audio samples.

![Preprocessing model(Figure 1)](https://github.com/Leon-Parepko/Audio-Generation/tree/main/Model.png)

To preprocess our data:

`python preprocessing/preprocessor_audioonly.py`

## Feature Extraction

We have developed a framework for extracting intermediate features in the form of a convolutional neural network. This innovative network not only processes the original audio sample, but also performs a detailed frequency analysis using Fast Fourier Transform (FFT) and mid-frequency cepstral coefficients (MFCC). These widespread methodologies are proven approaches for extracting comprehensive information from audio inputs.



## Postprocessing

The post processing stage aims to refine the generated voice output further, ensuring high-quality and natural-sounding results.
Applied techniques:

1) Waveform Smoothing
2) Pitch and Speed Normalization
3) Prosody Enhancement
4) Noise Reduction
5) Dynamic Range Compression


