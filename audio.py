import librosa
import numpy as np


class Audio:
    def __init__(self, path, sr=22050, duration=120):
        """
        Educated assumption: we can get all relevant info
        for techno song timbre in the first two minutes.
        """
        self.path = path
        self.y, self.sr = librosa.load(path, sr, duration)
        self.mfcc = None
        self.spec_con = None

    def extract_mfcc(self, n_mfcc=12):
        """
        Extract MFCC mean and std_dev vecs for an audio clip
        Save (2*n_mfcc,) shape (concatenated) vector in self.mfcc
        """
        mfcc = librosa.feature.mfcc(self.y, sr=self.sr, n_mfcc=n_mfcc).T
        mfcc_mean = mfcc.mean(axis=0)
        mfcc_std = mfcc.std(axis=0)
        self.mfcc = np.hstack([mfcc_mean, mfcc_std])

    def extract_spec_con(self):
        """
        Extract Spectral Contrast mean and std_dev vecs for an audio clip
        Save (2 * 7,) shape (concatenated) vector in self.spec_con
        """
        spec_con = librosa.feature.spectral_contrast(y=self.y, sr=self.sr).T
        spec_con_mean = spec_con.mean(axis=0)
        spec_con_std = spec_con.std(axis=0)
        self.spec_con = np.hstack([spec_con_mean, spec_con_std])


if __name__ == "__main__":
    pass
