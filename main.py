from audio import AudioFeature
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
    audio_features = [AudioFeature(metadata) for metadata in all_metadata]

    feature_matrix = np.vstack([audio.features for audio in audio_features])
    genre_labels = [audio.genre_label for audio in audio_features]
