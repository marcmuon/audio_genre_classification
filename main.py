from audio import AudioFeature
from model import Model
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder


def get_audio_metadata(playlist):
    """
    Currently assumes an Apple Music playlist saved as plain text.
    Returns: zip object with (audio_paths, genre_labels)
    """

    # TODO - assert check that length > intended offset+duration

    df = pd.read_csv(playlist, sep='\t')
    df = df[['Location', 'Genre']]

    paths = df['Location'].values.astype(str)
    paths = np.char.replace(paths, 'Macintosh HD', '')

    labels = df['Genre'].values

    return zip(paths, labels)


if __name__ == "__main__":

    all_metadata = get_audio_metadata(playlist='data/Music.txt')

    audio_features = []
    for metadata in all_metadata:
        audio = AudioFeature(metadata)
        audio.extract_features(save=True)
        audio_features.append(audio)

    # TODO - implement cache loader for data previously saved to disk

    feature_matrix = np.vstack([audio.features for audio in audio_features])
    genre_labels = [audio.genre_label for audio in audio_features]

    le = LabelEncoder()
    labels = le.fit_transform(genre_labels)

    model_cfg = dict(
        tt_dict=dict(shuffle=True, test_size=.2),
        scaler=StandardScaler(copy=True),
        base_model=RandomForestClassifier(random_state=42,
                                          n_jobs=4,
                                          class_weight='balanced',
                                          n_estimators=250,
                                          bootstrap=True),
        param_grid=dict(model__criterion=['entropy', 'gini'],
                        model__max_features=['log2', 'sqrt'],
                        model__min_samples_leaf=np.arange(2, 5)),
        grid_dict=dict(n_jobs=4,
                       refit=True,
                       iid=False),
        kf_dict=dict(n_splits=3,
                     random_state=42,
                     shuffle=True)
    )

    model = Model(feature_matrix, labels, model_cfg)
    model.run_cv_trials(n_trials=2)
    model.predict_from_holdout()

    # TODO need more jazz and ambient examples!
