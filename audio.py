import librosa
import numpy as np


class Audio:
    def __init__(self,
                 metadata,
                 duration=5,
                 offset=25,
                 sr=22050):
        """
        Keep duration num seconds of each clip, starting at
        offset num seconds into the song (avoid intros)
        """
        self.path, self.genre_label = metadata
        self.y, self.sr = librosa.load(self.path, sr, duration, offset)
        self.features = None
        self.local_path = None

    def _concat_features(self, feature):
        """
        Whenever an _extract_X method is called by main.py,
        this helper function concatatenates to Audio instance
        features attribute
        """
        self.features = np.hstack(
            [self.features, feature]
            if self.features is not None else feature)

    def extract_mfcc(self, n_mfcc=12):
        """
        Extract MFCC mean and std_dev vecs for a clip.
        Appends (2*n_mfcc,) shaped vector to
        instance feature vector
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
        Appends (2*(n_bands+1),) shaped vector to
        instance feature vector
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

    def save_local(self):
        self.local_path = self.path.split('/')[-1]
        np.save(f'data/{self.local_path}', self.features)


if __name__ == "__main__":
    pass
