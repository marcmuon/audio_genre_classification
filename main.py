from audio import Audio
import numpy as np
import pandas as pd


def get_audio_metadata(playlist):
    """
    Currently assumes an Apple Music playlist saved as plain text.
    Returns: zip object with (audio_paths, labels)
    """
    df = pd.read_csv(playlist, sep='\t')
    df = df[['Location', 'My Rating']]

    rating_map = {
        100.0: 5,
        80.0: 4,
        60.0: 3,
        40.0: 2,
        20.0: 1
    }
    df['My Rating'] = df['My Rating'].map(rating_map)

    if df['Location'][0].startswith('Macintosh HD'):
        df['Location'] = df['Location'].str.replace('Macintosh HD', '')

    paths = df['Location'].values
    labels = df['My Rating'].values

    return zip(paths, labels)


if __name__ == "__main__":

    all_metadata = get_audio_metadata(playlist='data/labeled_TEST.txt')
    audio_files = [Audio(metadata) for metadata in all_metadata]
    for audio in audio_files:
        audio.extract_features()  # populates audio.features attribute
    feature_matrix = np.vstack([audio.features for audio in audio_files])
    labels = [audio.label for audio in audio_files]
    dev_path = [audio.path for audio in audio_files]  # delete later

    # TODO - method to skip processing if path already exists in Audio
    # TODO - add save method for audio and delete it after extraction
