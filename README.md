# Plant Disease Detection using Convolutional Neural Networks (CNN)

## Overview
This project aims to develop a Convolutional Neural Network (CNN) model for the detection and classification of plant diseases based on leaf images. The model utilizes deep learning techniques to analyze images and accurately classify whether a plant is healthy or diseased, as well as identify the specific disease affecting the plant.

## Problem Statement
Plant diseases can significantly impact crop yield and quality, leading to economic losses for farmers. Traditional methods of disease detection are often time-consuming and may lack accuracy. By leveraging machine learning algorithms, we aim to automate the detection process and provide farmers with a reliable tool for early disease identification.

## Solution
The solution proposed in this project involves training a CNN model using a dataset of labeled leaf images, distinguishing between healthy plants and various disease types. The model architecture is designed to effectively extract features from input images and make accurate predictions regarding the presence of disease.
## Dataset 
Download the dataset from "https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset".

## Key Features
- Utilizes a CNN architecture for plant disease detection
- Trained on a dataset containing labeled leaf images
- Capable of identifying multiple disease types in various plant species
- Provides fast and accurate detection results

## Results
The trained CNN model achieved an accuracy of 91.46% on the validation dataset, with a testing loss of 0.2698. While the model demonstrates high performance, there are areas for improvement, such as handling variations in illumination conditions and complex backgrounds in input images.

## Future Scope
Future enhancements to the project may include:
- Refinement of the model architecture for improved performance
- Integration of advanced techniques such as Faster R-CNN or YOLO for precise disease localization
- Expansion of the dataset to include additional plant species and disease types
- Deployment of the model as a user-friendly application for farmers and agricultural practitioners

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- OpenCV

## Usage
1. Install the required dependencies using `pip install -r requirements.txt`.
2. Train the CNN model using the provided dataset and source code.
3. Evaluate the model's performance and adjust parameters as needed.
4. Deploy the trained model for real-world plant disease detection applications.
