
# Toxic Comment Classification with LSTM-CNN Hybrid Model
This project uses a hybrid model that combines Long Short-Term Memory (LSTM) and Convolutional Neural Network (CNN) architectures to classify toxic comments into six categories: toxic, severe toxic, obscene, threat, insult, and identity hate.

# Dataset
The dataset used in this project is the Toxic Comment Classification Challenge from Kaggle. It contains over 150,000 comments from Wikipedia that have been labeled by human raters for toxic behavior.

# Requirements
This project requires Python 3.x, TensorFlow 2.x, Keras, and other commonly used libraries such as NumPy and Pandas. The specific versions of these packages can be found in the requirements.txt file.

# Usage
To use the model, simply clone the repository and run train.py. This will download the dataset, preprocess the text data, and train the model. After training, the model will be saved to disk in the form of an HDF5 file.

You can then use the trained model to make predictions on new data using predict.py. This script takes in a text file containing comments, preprocesses the text, and predicts the toxicity levels of each comment using the trained model.

# Results
The LSTM-CNN hybrid model achieved an accuracy of 98% on the test dataset, which outperformed several state-of-the-art models in the literature.

#Acknowledgements
This project was completed as part of a Kaggle competition and was inspired by several open-source projects available on GitHub. We would like to acknowledge the Kaggle community and the authors of these projects for their contributions.
