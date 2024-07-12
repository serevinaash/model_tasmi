import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, Input, Bidirectional, BatchNormalization, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
import joblib

# Define the dataset paths
dataset_paths = [
    'C:/Users/MSI/Downloads/Ghamadi_40kbps-20240710T015805Z-001/Ghamadi_40kbps'
]

# Load transcripts from the uploaded file
transcripts_path = 'C:/Users/MSI/Downloads/modified_transcripts (1).tsv'
transcripts = pd.read_csv(transcripts_path, sep='\t')

# Replace placeholder with actual path in transcripts
for dataset_path in dataset_paths:
    transcripts['PATH'] = transcripts['PATH'].str.replace('${DATASET_PATH}', dataset_path)

# Load audio files
audio_files = []
for dataset_path in dataset_paths:
    audio_files.extend([os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.endswith('.mp3')])
audio_files = sorted(audio_files)

# Ensure transcripts match the number of audio files
if len(transcripts) != len(audio_files):
    raise ValueError(f"Number of transcripts ({len(transcripts)}) does not match number of audio files ({len(audio_files)})")

# Take the first 30 audio files and transcripts
subset_audio_files = audio_files[:30]
subset_transcripts = transcripts.iloc[:30]

# Feature extraction function
def extract_features(file_path, augment=False):
    try:
        audio, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    if augment:
        # Apply augmentation techniques
        if np.random.rand() < 0.5:
            audio = audio + 0.005 * np.random.randn(len(audio))  # Add white noise
        if np.random.rand() < 0.5:
            audio = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.8, 1.2))  # Time stretching
        if np.random.rand() < 0.5:
            audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=np.random.randint(-3, 3))  # Pitch shifting

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    combined = np.concatenate((mfccs, delta_mfccs, delta2_mfccs), axis=0)
    return combined.T

# Extract features and align with transcripts
features = []
labels = []
for audio_file, transcript in zip(subset_audio_files, subset_transcripts['TRANSCRIPT']):
    mfccs = extract_features(audio_file)
    if mfccs is not None:
        features.append(mfccs)
        labels.append(transcript)

        # Augmented data
        mfccs_aug = extract_features(audio_file, augment=True)
        if mfccs_aug is not None:
            features.append(mfccs_aug)
            labels.append(transcript)

# Pad sequences to ensure equal length
max_seq_length = max([feature.shape[0] for feature in features])
padded_features = pad_sequences(features, maxlen=max_seq_length, padding='post', dtype='float32')

# Normalize features
mean_features = np.mean(padded_features, axis=0)
std_features = np.std(padded_features, axis=0)
padded_features = (padded_features - mean_features) / std_features

# Encode labels as integers
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)

# Save LabelEncoder
label_encoder_path = 'C:/Users/MSI/Downloads/label_encoder.pkl'
joblib.dump(label_encoder, label_encoder_path)
print(f'LabelEncoder saved as {label_encoder_path}')

# One-hot encode the integer labels
onehot_encoded = to_categorical(integer_encoded)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_features, onehot_encoded, test_size=0.3, random_state=42)

# Define the model
model = Sequential([
    Input(shape=(max_seq_length, 39)),
    Masking(mask_value=0.0),
    Bidirectional(LSTM(512, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))),
    BatchNormalization(),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(512, kernel_regularizer=tf.keras.regularizers.l2(0.01))),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

# Add callbacks for model checkpoint, learning rate reduction, and early stopping
callbacks = [
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=512, validation_data=(X_val, y_val), callbacks=callbacks)

# Save the trained model
model.save('C:/Users/MSI/Downloads/trained_model.h5')
print('Model saved as trained_model.h5')

# Visualize training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

# Inference: Read audio file and predict transcript
def predict_transcript(audio_file_path, model, label_encoder, mean_features, std_features):
    mfccs = extract_features(audio_file_path)
    if mfccs is None:
        return None

    padded_mfccs = pad_sequences([mfccs], maxlen=max_seq_length, padding='post', dtype='float32')
    normalized_mfccs = (padded_mfccs - mean_features) / std_features

    prediction = model.predict(normalized_mfccs)
    predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=1))
    return predicted_label[0]

# Load the saved model for inference
loaded_model = tf.keras.models.load_model('C:/Users/MSI/Downloads/trained_model.h5')
print('Model loaded from trained_model.h5')

# Load the saved LabelEncoder
loaded_label_encoder = joblib.load(label_encoder_path)
print('LabelEncoder loaded from label_encoder.pkl')

# Test the function with an example audio file
example_audio_file = subset_audio_files[-1]  # Use the last audio file for prediction
predicted_transcript = predict_transcript(example_audio_file, loaded_model, loaded_label_encoder, mean_features, std_features)
print(f'Predicted transcript for {example_audio_file}: {predicted_transcript}')

# Test the function with another example audio file to ensure the model can predict multiple times
another_example_audio_file = subset_audio_files[-2]  # Use another audio file for prediction
another_predicted_transcript = predict_transcript(another_example_audio_file, loaded_model, loaded_label_encoder, mean_features, std_features)
print(f'Predicted transcript for {another_example_audio_file}: {another_predicted_transcript}')