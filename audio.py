from librosa import feature, beat, load as librosa_load
import numpy as np


class Audio:
    def __init__(self, path, sr=22050, duration=120):
        """
        Educated assumption: we can get all relevant info
        for techno song timbre in the first two minutes.
        """
        self.path = path
        self.y, self.sr = librosa_load(path, sr, duration)
        self.mfcc = None
        self.spectral = None
        self.tempo = None

    def extract_mfcc(self, n_mfcc=12):
        """
        Extract MFCC mean and std_dev vecs for an audio clip.
        Save (2*n_mfcc,) shaped (concatenated) vector in self.mfcc
        """
        mfcc = feature.mfcc(self.y, sr=self.sr, n_mfcc=n_mfcc)
        mfcc_mean = mfcc.mean(axis=1).T
        mfcc_std = mfcc.std(axis=1).T
        self.mfcc = np.hstack([mfcc_mean, mfcc_std])

    def extract_spectral(self, n_bands=3):
        """
        Extract Spectral Contrast mean and std_dev vecs for an audio clip.
        Save (2 * n_bands+1,) shaped (concatenated) vector in self.spectral
        """
        spec_con = feature.spectral_contrast(y=self.y, sr=self.sr, n_bands=3)
        spec_con_mean = spec_con.mean(axis=1).T
        spec_con_std = spec_con.std(axis=1).T
        self.spectral = np.hstack([spec_con_mean, spec_con_std])

    def extract_tempo(self):
        """
        Extract the BPM in (1,) shaped matrix
        """
        self.tempo = beat.tempo(y=self.y, sr=self.sr)


if __name__ == "__main__":
    pass
