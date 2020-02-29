import librosa


class Audio:
    def __init__(self, path, sr=22050, duration=120):
        """
        Educated assumption: we can get all relevant info
        for techno song timbre in the first two minutes.
        """
        self.path = path
        self.y, self.sr = librosa.load(path, sr, duration)


if __name__ == "__main__":
    pass
