import numpy as np
from scipy.io import wavfile

class FFTPreprocessor:
    def __init__(self, rate):
        self.RATE = rate
        self.MAX_FRAME = int(self.RATE * 30)
        self.MIN_FRAME = int(self.RATE * 0.3)
        self.MAX_INPUT = int(self.MAX_FRAME / self.MIN_FRAME)
        self.FREQUENCY_BINS = int(self.MIN_FRAME / 2) + 1
        self.NORM_FACTOR = 1.0 / 2 ** 16.0

    def make_tensor(self, path):
        rate, data = wavfile.read(path)
        data = np.array([(e * self.NORM_FACTOR) * 2 for e in data])
        output = nn_input = np.zeros((self.FREQUENCY_BINS,
                                      self.MAX_INPUT,
                                      2))
        freqs, times, specs = signal.spectrogram(data,
                                                 fs=self.RATE,
                                                 window="boxcar",
                                                 nperseg=self.MIN_FRAME,
                                                 noverlap=0,
                                                 detrend=False,
                                                 mode='complex')
        output[:, :specs.shape[1], 0] = np.real(specs)
        output[:, :specs.shape[1], 1] = np.imag(specs)
        return output