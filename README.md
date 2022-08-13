<center><h1>A CNN Based Air-Writing Recognition Framework for Linguistic Characters</h1></center>

Air writing is a practice of writing the linguistic characters in free space utilizing the six degrees of freedom of hand motion. We propose a system that uses a generic webcam to detect and recognize the virtually written characters by a user as per their will. This system performs detection using HSV color space for creating the masking of the tracker or the tracking object and morphological operations for refinement of the mask. This system gives the user the freedom to select a writing object of any color, shape, or material for tracking purposes. The trajectory of the contour of the object’s mask is tracked and rendered on a virtual window. The air-written character is recognized using the Convolutional Neural Network (CNN). The CNN is trained on four different datasets, which are English handwritten characters of 26 different classes (A-Z), MNIST dataset with 10 different classes (0-9), Devanagari handwritten character dataset consisting of 36 different classes (ka-gya), and Devanagari handwritten digits consisting of 10 classes (0-9). The accuracy achieved by the proposed system for isolated characters on respective datasets is 99.75%, 99.73%, 99.13%, and 99.97%.

The dataset used are [A_Z Handwritten](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format), [MNIST](https://www.kaggle.com/competitions/digit-recognizer/data), [Devanagari Characters](https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset), [Devanagari Digits](https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset).


For training, put the following script in the terminal:

`python train.py True <dataset_name>` 

For evaluating the exisitng model, put the following script in the terminal:

`python train.py False <dataset_name>`
