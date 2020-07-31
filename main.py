from audio import AudioFeature
from model import Model
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os
import pickle


def parse_audio_playlist(playlist):
    """
    Assumes an Apple Music playlist saved as plain text as parse input.
    Returns: zip object with (paths, genres)
    """

    df = pd.read_csv(playlist, sep="\t")
    df = df[["Location", "Genre"]]

    paths = df["Location"].values.astype(str)
    paths = np.char.replace(paths, "Macintosh HD", "")

    genres = df["Genre"].values

    return zip(paths, genres)
        

def load_saved_audio_features(path):
    
    files = next(os.walk(path))
    pkl_list = files[-1]
    pkl_list = [p for p in pkl_list if p.endswith(".pkl")]
    
    audio_features = []
    
    for p in pkl_list:
        with open(f"{path}{p}", "rb") as input_file:
            e = pickle.load(input_file)
            audio_features.append(e)
    
    return audio_features


if __name__ == "__main__":

    all_metadata = parse_audio_playlist(playlist="data/Subset.txt")

    audio_features = []
    for metadata in all_metadata:
        path, genre = metadata
        audio = AudioFeature(path, genre)
        audio.extract_features("mfcc", "spectral_contrast", "tempo", save_local=True)
        audio_features.append(audio)

    # audio_features = load_saved_audio_features("./data/")

    feature_matrix = np.vstack([audio.features for audio in audio_features])
    genre_labels = [audio.genre for audio in audio_features]

    model_cfg = dict(
        tt_test_dict=dict(shuffle=True, test_size=0.3),
        tt_val_dict=dict(shuffle=True, test_size=0.25),
        scaler=StandardScaler(copy=True),
        base_model=RandomForestClassifier(
            random_state=42,
            n_jobs=4,
            class_weight="balanced",
            n_estimators=250,
            bootstrap=True,
        ),
        param_grid=dict(
            model__criterion=["entropy", "gini"],
            model__max_features=["log2", "sqrt"],
            model__min_samples_leaf=np.arange(2, 4),
        ),
        grid_dict=dict(n_jobs=4, refit=True, iid=False, scoring="balanced_accuracy"),
        kf_dict=dict(n_splits=3, random_state=42, shuffle=True),
    )

    model = Model(feature_matrix, genre_labels, model_cfg)
    model.train_kfold()
    model.predict(holdout_type="val")
    model.predict(holdout_type="test")

    