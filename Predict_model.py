import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import joblib

# Define the path to the best model
model_path = '/content/trained_model.h5'

# Load the trained model
loaded_model = load_model(model_path)
print('Model loaded from best_model.keras')

# Define a function to predict transcript from an audio file
def predict_transcript(audio_file_path, model, label_encoder):
    # Define the feature extraction function
    def extract_features(file_path):
        try:
            audio, sr = librosa.load(file_path, sr=None)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        delta_mfccs = librosa.feature.delta(mfccs)
        combined = np.concatenate((mfccs, delta_mfccs), axis=0)
        return combined.T

    # Extract features from the audio file
    mfccs = extract_features(audio_file_path)
    if mfccs is None:
        return None

    # Pad sequences to match model input shape
    max_seq_length = model.layers[0].input_shape[1]
    padded_mfccs = pad_sequences([mfccs], maxlen=max_seq_length, padding='post', dtype='float32')

    # Normalize features, handling potential division by zero
    mean = np.mean(padded_mfccs, axis=0)
    std = np.std(padded_mfccs, axis=0)
    std[std == 0] = 1.0  # Replace zeros with 1 to avoid division by zero
    padded_features = (padded_mfccs - mean) / std

    # Check for NaN values after normalization
    if np.isnan(np.sum(padded_features)):
        print(f"NaN values encountered after normalization for {audio_file_path}. Skipping prediction.")
        return None

    # Predict transcript
    prediction = model.predict(padded_features)
    predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=1))

    return predicted_label[0]

# Define the path to the audio file you want to predict
input_audio_file = '/content/001004.mp3'

# Load label encoder used during training
label_encoder_path = '/content/label_encoder.pkl'
label_encoder = joblib.load(label_encoder_path)

# Perform prediction
predicted_transcript = predict_transcript(input_audio_file, loaded_model, label_encoder)
print(f'Predicted transcript for {input_audio_file}: {predicted_transcript}')
