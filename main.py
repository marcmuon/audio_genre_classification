from os.path import expanduser
from audio import Audio
import os


def get_audio_paths(audio_dir):
    """
    Place directory containing audio in your home dir.
    """
    home = expanduser("~")
    audio_dir = os.path.join(home, audio_dir)
    full_paths = []
    for file in os.listdir(audio_dir):
        if not file.endswith(".DS_Store"):
            full_paths.append(os.path.join(audio_dir, file))
    return full_paths


if __name__ == "__main__":
    paths = get_audio_paths('data/audio')
    audio_list = [Audio(path) for path in paths]
    for audio in audio_list:
        audio.extract_mfcc()
        audio.extract_spec_con()
