# Image Captioning Project

This project implements a deep learning model to automatically generate captions for images. It serves as a practical application of computer vision and natural language processing techniques, and can be a valuable visual aid for individuals with visual impairments.

## Project Overview

The core of this project is a model that takes an image as input and produces a descriptive, human-readable sentence as output. This is achieved by combining a pre-trained convolutional neural network (CNN) for image feature extraction with a recurrent neural network (RNN) for sequence generation.

### Key Features:

* **Image Feature Extraction:** Utilizes the ResNet50 architecture, pre-trained on the ImageNet dataset, to extract meaningful feature vectors from images. The final classification layer of ResNet50 is removed to leverage the rich, high-level feature representations.
* **Text Processing:** Employs natural language processing techniques to clean and prepare the textual captions for training. This includes converting text to lowercase, removing punctuation and numbers, and creating a vocabulary of unique words.
* **Word Embeddings:** Uses pre-trained GloVe (Global Vectors for Word Representation) embeddings to convert words into dense vector representations, capturing semantic relationships between them. The model uses the 50-dimensional GloVe embeddings.
* **Model Architecture:** The model consists of two main components:
    * An **image embedding** part, which processes the extracted image features through a Dense layer.
    * A **language model** part, which uses an Embedding layer (initialized with GloVe weights) and an LSTM (Long Short-Term Memory) network to process the sequence of words in the caption.
    * These two components are merged and passed through further Dense layers to predict the next word in the sequence.
* **Training and Prediction:** The model is trained to predict the next word in a caption given the image features and the preceding words. During inference, it generates a caption word by word until an end-of-sequence token is produced.

## How It Works

1.  **Data Loading and Preprocessing:**
    * Image captions are loaded from `captions.txt`.
    * The text is cleaned by converting it to lowercase, removing punctuation and numbers, and filtering out short words.
    * A vocabulary is built from the cleaned captions, and words with a frequency below a certain threshold (10 in this notebook) are excluded.
    * "startseq" and "endseq" tokens are added to each caption to signify the beginning and end of a sentence.

2.  **Image Feature Extraction:**
    * The ResNet50 model is used to extract a 2048-dimensional feature vector for each image in the training and testing datasets.
    * These feature vectors are saved as `.pkl` files for efficient loading during training.

3.  **Model Training:**
    * The model is trained using a data generator that provides batches of data. Each batch consists of an image feature vector, a partial input sequence of words, and the corresponding next word to be predicted.
    * The model uses the Adam optimizer and is compiled with categorical cross-entropy loss.
    * The model is trained for 20 epochs, with weights being saved after each epoch.

4.  **Inference/Prediction:**
    * For a given test image, its pre-computed feature vector is fed into the model.
    * Starting with the "startseq" token, the model predicts the next word in the caption.
    * This predicted word is then used as input for the next time step, and this process continues until the "endseq" token is generated or the maximum caption length is reached.

## Files in this Repository

* `Image_Captioning.ipynb`: The main Jupyter Notebook containing the complete code for data preprocessing, model building, training, and prediction.
* `dataset/`: This directory should contain the image files and text files with captions.
    * `images/`: Directory for all the image files.
    * `captions.txt`: The raw captions file.
    * `train.txt` & `test.txt`: Files listing the image names for the training and testing sets.
* `saved/`: This directory should contain the pre-trained GloVe embeddings (`glove.6B.50d.txt`).
* `model_weights/`: This directory will store the trained model weights after each epoch.
* `encoded_train_features.pkl` & `encoded_test_features.pkl`: Pickle files containing the extracted image features for the training and testing sets.

## How to Use

1.  **Prerequisites:** Ensure you have the necessary Python libraries installed, including `pandas`, `numpy`, `matplotlib`, `keras`, `nltk`, and `opencv-python`.
2.  **Dataset:** Download the required dataset (e.g., Flickr8k) and place the images and caption files in the `dataset/` directory.
3.  **GloVe Embeddings:** Download the GloVe embeddings and place the `glove.6B.50d.txt` file in the `saved/` directory.
4.  **Run the Notebook:** Execute the cells in the `Image_Captioning.ipynb` notebook sequentially to preprocess the data, build and train the model, and generate captions for new images.
