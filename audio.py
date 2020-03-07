import librosa
import numpy as np


class Audio:
    def __init__(self, metadata, sr=22050, duration=120):
        """
        Educated assumption: we can get all relevant info
        for techno song timbre in the first two minutes.
        """
        self.path, self.label = metadata
        self.y, self.sr = librosa.load(self.path, sr, duration)
        self.features = None

    def _concat_features(self, feature):
        self.features = np.hstack(
            [self.features, feature]
            if self.features is not None else feature)

    def extract_mfcc(self, n_mfcc=12):
        """
        Extract MFCC mean and std_dev vecs for a clip.
        Appends (2*n_mfcc,) shaped vector to self.features
        """
        mfcc = librosa.feature.mfcc(self.y,
                                    sr=self.sr,
                                    n_mfcc=n_mfcc)

        mfcc_mean = mfcc.mean(axis=1).T
        mfcc_std = mfcc.std(axis=1).T
        mfcc_feature = np.hstack([mfcc_mean, mfcc_std])
        self._concat_features(mfcc_feature)

    def extract_spectral_contrast(self, n_bands=3):
        """
        Extract Spectral Contrast mean and std_dev vecs for a clip.
        Appends (2*(n_bands+1),) shaped vector to self.features
        """
        spec_con = librosa.feature.spectral_contrast(y=self.y,
                                                     sr=self.sr,
                                                     n_bands=3)

        spec_con_mean = spec_con.mean(axis=1).T
        spec_con_std = spec_con.std(axis=1).T
        spec_con_feature = np.hstack([spec_con_mean, spec_con_std])
        self._concat_features(spec_con_feature)

    def extract_tempo(self):
        """
        Extract the BPM.
        Appends (1,) shaped vector to instance feature vector
        """
        tempo = librosa.beat.tempo(y=self.y, sr=self.sr)
        self._concat_features(tempo)


if __name__ == "__main__":
    pass
