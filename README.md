# Concrete Crack Image Classifier

## Description
This repository contains code for Concrete Crack Image Classification. The code is designed to classify images of concrete cracks into different categories using transfer learning.

## Installation
To use this code, please follow the steps below:

1. Clone the repository to your local machine using the following command

```
git clone [repository URL]
```

2. Navigate to the project directory:

```
cd [project directory]
```

3. Install the required dependencies by running the following command:

```
pip install pandas numpy tensorflow matplotlib scikit-learn
```

4. Download the dataset from the provided URL in **Credits** and place it in the appropriate location as specified in the code.

## Usage
1. Run the Jupyter notebook or Python script to execute the code.
2. The code performs the following steps:
- Splits the dataset into training, validation, and testing sets.
- Loads the image data using the TensorFlow image_dataset_from_directory function.
- Performs data inspection by visualizing some example images.
- Converts the dataset into a prefetch dataset for improved performance.
- Applies data augmentation techniques using the Keras Sequential model.
- Performs pixel standardization on the images.
- Applies transfer learning using the MobileNetV2 model.
- Compiles and trains the model using the specified optimizer, loss function, and metrics.
- Evaluates the model's performance on the test set.
- Deploys the model by predicting the classes of a batch of test images.
- Saves the trained model in the .h5 format for future use.
3. Customize the code as needed, such as adjusting the hyperparameters, modifying the data augmentation techniques, or changing the architecture of the model.

## Outputs

- model training
![model training](https://github.com/FIT003/YPAI03_ConcreteCrack_Classifier/assets/97938451/0b4c34e6-61a6-4dba-9d9b-0e11467f7f07)

- test accuracy
![test accuracy](https://github.com/FIT003/YPAI03_ConcreteCrack_Classifier/assets/97938451/803fe3a0-3f63-47ec-a3d2-6944c7a77e83)

- loss
![epoch_loss](https://github.com/FIT003/YPAI03_ConcreteCrack_Classifier/assets/97938451/72b9d351-1c9c-4ad6-abe6-20871a2193ed)

- loss vs iteration
![loss_vs_iteration](https://github.com/FIT003/YPAI03_ConcreteCrack_Classifier/assets/97938451/904f6478-7631-451b-a5d9-0078b35b631f)

- accuracy
![epoch_accuracy](https://github.com/FIT003/YPAI03_ConcreteCrack_Classifier/assets/97938451/c5f001fe-a982-40f1-a931-59b9851955fc)

- accuracy vs iteration
![accuracy_vs_iteration](https://github.com/FIT003/YPAI03_ConcreteCrack_Classifier/assets/97938451/3b41568c-6c43-45de-b18d-adf64c35736e)


## Credit
URL: https://data.mendeley.com/datasets/5y9wdsg2zt/2
