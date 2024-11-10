import os
import numpy as np
import tensorflow as tf
import librosa
import seaborn as sns
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing import image
from matplotlib.patches import Rectangle

# Define the classes for classification
classes = ["car_horn", "cat", "dog", "glass_breaking", "siren"]

# Load the trained model (ensure it's the correct model path)
model = tf.keras.models.load_model('classifier_model.keras')
class_images_dir = "./class_images"  # Defining directory containing class images

# Load YAMNet model
yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(yamnet_model_handle)

# Function to load and preprocess audio
def load_and_preprocess_audio(file_path, target_sr=16000):
    file_contents = tf.io.read_file(file_path)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1) # Converting to wav form and setting to mono channel
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int32)
    wav = librosa.resample(wav.numpy(), orig_sr=sample_rate.numpy(), target_sr=target_sr) # Converting to form that is accepted by YAMNet
    return tf.convert_to_tensor(wav, dtype=tf.float32)

# Function to extract YAMNet embeddings from the audio
def extract_yamnet_embeddings(audio):
    audio = tf.squeeze(audio)
    _, embeddings, _ = yamnet_model(audio)  # Only the embeddings are used
    avg_embedding = tf.reduce_mean(embeddings, axis=0)  # Average across time
    return avg_embedding.numpy()  # Converting to numpy array for model input

# Function to classify an individual audio file and return prediction and confidence values
def classify_file(file_path):
    # Load and preprocess the audio file
    audio_features = load_and_preprocess_audio(file_path)
    audio_features = np.expand_dims(audio_features, axis=0)  # Add batch dimension
    
    # Extract YAMNet embeddings
    embedding = extract_yamnet_embeddings(audio_features)
    embedding = np.expand_dims(embedding, axis=0)  # Add batch dimension
    
    # Make the prediction
    predictions = model.predict(embedding)

    # Get the class index with highest probability
    predicted_class_index = np.argmax(predictions, axis=1)
    predicted_class = classes[predicted_class_index[0]]

    # Return the predicted class and the confidence values
    return predicted_class, predictions[0], predicted_class_index[0]

# Function to classify and plot a single file with confidence values
def classify_and_plot_file(file_path):
    predicted_class, confidence_values, predicted_class_index = classify_file(file_path=file_path)

    # Load the image for the predicted class
    class_image_path = os.path.join(class_images_dir, f"{predicted_class}.jpg")
    img = image.load_img(class_image_path, target_size=(300, 300))  # Resize image to fit in the plot
    img_array = image.img_to_array(img)  # Convert image to array
    
    # Plot the image and the bar chart of confidence values
    _, ax = plt.subplots(1, 2, figsize=(10, 5))  # Create a figure with 2 subplots
    
    # Add a border around the image
    ax[0].imshow(img_array.astype('uint8'))
    ax[0].axis('off')  # Turn off axis

    # Keep the aspect ratio square
    ax[0].set_aspect('equal', adjustable='box')
    
    # Add border around image
    border = Rectangle((0, 0), 1, 1, linewidth=10, edgecolor='black', facecolor='none')
    ax[0].add_patch(border)
    
    # Add text below the image
    ax[0].text(0.5, -0.1, f"Predicted class: {predicted_class}", ha="center", va="center", 
               transform=ax[0].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot bar chart on the right side
    ax[1].bar(classes, confidence_values)  # Plot all classes' confidence values
    ax[1].set_title(f"Prediction: {predicted_class} (Confidence: {confidence_values[predicted_class_index]*100:.2f}%)")
    ax[1].set_ylabel('Confidence')
    ax[1].set_ylim([0, 1])  # Set y-axis from 0 to 1
    
    # Show the plot
    plt.tight_layout(pad=4.0)
    plt.show()
    
    return predicted_class, confidence_values

# Function to classify all audio files in a directory and plot confidence values for each
def classify_directory(data_dir):
    true_labels = []
    predicted_labels = []
    
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                if filename.endswith('.wav'):
                    file_path = os.path.join(class_dir, filename)
                    # Classify and plot the result for each file
                    predicted_class, _, _ = classify_file(file_path)
                    
                    # Store the result
                    true_labels.append(class_name)
                    predicted_labels.append(predicted_class)
                    
    # Return the true labels and predicted labels
    return true_labels, predicted_labels

# Function to classify and plot the entire directory
def classify_and_plot_directory(directory_path):
    # Example usage for classifying all files in a directory:
    true_labels, predicted_labels = classify_directory(directory_path)

    # Generate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)

    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
 
testing_data_dir = "./testing_data"

# Main function to parse command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Audio Classification")
    parser.add_argument('--file', type=str, help="Path to the audio file for individual classification")
    args = parser.parse_args()
    
    if args.file:
        print(f"Classifying individual file: {args.file}")
        predicted_class, confidence_values = classify_and_plot_file(args.file)
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence Values: {confidence_values}")
    else:
        classify_and_plot_directory(testing_data_dir)

if __name__ == "__main__":
    main()