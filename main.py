from audio import Audio
import numpy as np
import pandas as pd


def get_audio_metadata(playlist):
    """
    Currently assumes an Apple Music playlist saved as plain text.
    Returns: zip object with (audio_paths, genre_labels)
    """
    df = pd.read_csv(playlist, sep='\t')
    df = df[['Location', 'Genre']]

    paths = df['Location'].values.astype(str)
    paths = np.char.replace(paths, 'Macintosh HD', '')

    labels = df['Genre'].values

    return zip(paths, labels)


if __name__ == "__main__":

    all_metadata = get_audio_metadata(playlist='data/Music.txt')
    audio_files = [Audio(metadata) for metadata in all_metadata]

    for audio in audio_files:  # populates audio.features in each instance
        audio.extract_mfcc()
        audio.extract_spectral_contrast()
        audio.extract_tempo()
        audio.save_local()

    feature_matrix = np.vstack([audio.features for audio in audio_files])
    genre_labels = [audio.genre_label for audio in audio_files]
