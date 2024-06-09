import os
import numpy as np
import librosa

def extract_features(file_name):
    y, sr = librosa.load(file_name)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def load_data(data_dir):
    labels = []
    features = []
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                features.append(extract_features(file_path))
                labels.append(label)
    return np.array(features), np.array(labels)

# Örneğin veri klasörü yapısı:
# data/
#   - speaker1/
#       - audio1.wav
#       - audio2.wav
#   - speaker2/
#       - audio1.wav
#       - audio2.wav

data_dir = 'data'
features, labels = load_data(data_dir)
