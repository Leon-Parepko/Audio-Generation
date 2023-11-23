from FFT import FFTPreprocessor
from NN import NN
import librossa
import sklearn
import torch
import numpy as np

class FeatureExtractor:
    def __init__(self, rate, num_features):
        self.rate = rate
        self.FFT = FFTPreprocessor(rate)
        self.model = NN(num_features)

        # load model weights
        self.model.load_state_dict(torch.load('feature_extractor_NN.pth'))
        self.model.eval()


    def forward(self, path):
        # Get the mfccs
        data, sr = librosa.load(path)
        mfccs = librosa.feature.mfcc(data, sr=sr)
        mfccs = sklearn.preprocessing.scale(mfccs, axis=1)

        # Get the FFT
        fft = self.FFT.make_tensor(path)

        # Get the chroma
        chroma = librosa.feature.chroma_stft(data, sr=sr)
        chroma = sklearn.preprocessing.scale(chroma, axis=1)

        # merge all the data to pass to the model
        data = np.concatenate((mfccs, fft, chroma), axis=0)
        data = torch.from_numpy(data).float()

        # forward the neural network
        output = self.model(data.unsqueeze(0))
        return output


