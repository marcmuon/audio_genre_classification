from audio import Audio
import numpy as np
import pandas as pd


def get_audio_metadata(playlist):
    """
    Assumes an Apple Music playlist saved as plain text.
    """
    df = pd.read_csv(playlist, sep='\t')
    df = df[['Location', 'My Rating']]
    df['My Rating'] = df['My Rating'].map({
        100.0: 5,
        80.0: 4,
        60.0: 3,
        40.0: 2,
        20.0: 1})

    if df['Location'][0].startswith('Macintosh HD'):
        df['Location'] = df['Location'].str.replace('Macintosh HD', '')

    paths = df['Location'].values
    labels = df['My Rating'].values
    metadata = zip(paths, labels)

    return metadata


if __name__ == "__main__":

    all_metadata = get_audio_metadata(playlist='data/labeled_TEST.txt')
    audio_librosa = [Audio(metadata) for metadata in all_metadata]
    for audio in audio_librosa:
        # Calling these methods populate 'audio.features'
        audio.extract_mfcc()
        audio.extract_spectral_contrast()
        audio.extract_tempo()
    feature_matrix = np.vstack([audio.features for audio in audio_librosa])
