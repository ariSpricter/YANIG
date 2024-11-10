import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load YAMNet model from TensorFlow Hub
yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(yamnet_model_handle)

# Function to load and preprocess audio for MFCC feature extraction
def load_and_preprocess_audio(file_path, target_sr=16000):
    # Load and resample audio using librosa
    file_contents = tf.io.read_file(file_path)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int32)

    wav = librosa.resample(wav.numpy(), orig_sr=sample_rate.numpy(), target_sr=target_sr)
    return tf.convert_to_tensor(wav, dtype=tf.float32)

# Function to extract averaged YAMNet embeddings for each audio file
def extract_yamnet_embeddings(audio):
    _, embeddings, _ = yamnet_model(audio)  # Only embeddings are used
    # Average across all embeddings to get a single 1024-dimensional vector
    avg_embedding = tf.reduce_mean(embeddings, axis=0)
    return avg_embedding

# Load and preprocess all audio files in your dataset
def load_data(data_dir, classes):
    data = []
    labels = []
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(class_dir, filename)
                audio = load_and_preprocess_audio(file_path)
                embedding = extract_yamnet_embeddings(audio)
                data.append(embedding.numpy())  # Convert Tensor to NumPy array
                labels.append(i)
    return np.array(data), np.array(labels)

# Set your data directory and classes
data_dir = "./training_data"  # Replace with actual data directory
classes = ["car_horn", "cat", "dog", "glass_breaking", "siren"]  # Replace with your class names

# Load the data and labels, and one-hot encode the labels
data, labels = load_data(data_dir, classes)
labels = to_categorical(labels, num_classes=len(classes))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Best hyperparameters from your previous tuning
best_num_layers = 1   # 1 hidden layer
best_units_0 = 256    # 256 units in the hidden layer
best_learning_rate = 0.0006737447552989427   # Learning rate
best_batch_size = 32  # Best batch size found
best_epochs = 20      # Best number of epochs (from your tuning)

# Define the new model that uses YAMNet embeddings as input, based on the best hyperparameters
model = Sequential([
    Dense(best_units_0, activation='relu', input_shape=(1024,)),  # 1024 = YAMNet embedding size
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=best_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Set up early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=best_epochs, batch_size=best_batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Plot accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Loss: {test_loss * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

saved_model_path = 'classifier_model.keras'

"""Comment this line to stop saving new models"""
model.save(saved_model_path, include_optimizer=False)

print("Model saved for deployment.")