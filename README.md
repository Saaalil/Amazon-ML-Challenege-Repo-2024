# Product Attribute Prediction Model

## Overview

This project implements a deep learning model using PyTorch to predict product attributes from images. The model is designed to predict both `group_id` and `entity_name` attributes for each image, leveraging a pre-trained ResNet-50 model as the backbone for feature extraction. The dataset includes product images, corresponding group IDs, and entity names.

## Dataset

The dataset is expected to have the following format in a CSV file (`train.csv`):

| image_link         | group_id | entity_name | entity_value |
|--------------------|----------|-------------|--------------|
| /path/to/image.jpg | group1   | entity1     | value1       |

### File Paths:
- **CSV file**: Path to the dataset file (default: `train.csv`)
- **Image directory**: Path to the folder containing images
- **Output file**: Path to save predictions (default: `predictions.csv`)

## Model Architecture

The model uses a pre-trained ResNet-50 model to extract features from the input images. It adds two fully connected layers for predicting `group_id` and `entity_name`:

- **ResNet-50 backbone**: Used for feature extraction.
- **Two fully connected layers**: One for predicting the group ID and the other for predicting the entity name.

## Training and Prediction

1. **Training**: The model is trained using cross-entropy loss for both group ID and entity name predictions. It utilizes the Adam optimizer with gradient scaling to handle mixed precision on GPUs.
   
2. **Prediction**: After training, the model predicts both `group_id` and `entity_name` for each image and saves the results to a CSV file.

## Code Structure

### `ProductImageDataset` Class

Custom `Dataset` class to load images and their corresponding attributes from the CSV file. It applies the necessary transformations, including resizing and normalization, to the images.

### `ProductAttributeModel` Class

Defines the architecture of the model, with ResNet-50 as the base and two additional fully connected layers for attribute prediction.

### Functions

- **`load_dataset(csv_path, limit=None)`**: Loads the dataset from the CSV file and preprocesses it. It also limits the dataset size for faster processing.
  
- **`train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS)`**: Handles the training loop, including calculating losses and accuracy for both training and validation sets.

- **`predict_and_save(model, test_loader, group_ids, entity_names, output_csv_path)`**: Uses the trained model to predict attributes for the test set and saves the results in a CSV file.

### `main()` Function

The main function that loads the dataset, trains the model, and saves the predictions. It performs the following steps:
1. Load the dataset.
2. Create PyTorch `DataLoader` for training and validation data.
3. Initialize the model, loss function, and optimizer.
4. Train the model and save the best model's state.
5. Predict the attributes for the validation set and save the predictions.

## How to Run

1. **Set up dependencies**: Install the required packages.
   ```bash
   pip install torch torchvision tqdm Pillow scikit-learn
   ```

2. **Prepare your dataset**: Ensure the CSV file and image directory are properly set up.

3. **Run the code**: Execute the `main()` function to train the model and generate predictions.
   ```bash
   python product_attribute_model.py
   ```

## Model Checkpoints

The model checkpoint will be saved in the current directory as `product_attribute_model.pth`.

## Output

- **Predictions**: A CSV file `predictions.csv` will be generated with the predicted group IDs and entity names for each product.
  
  | predicted_group_id | predicted_entity_name | entity_value |
  |--------------------|-----------------------|--------------|
  | group1             | entity1               | value1       |

## Parameters

- **CSV_PATH**: Path to the CSV file with the dataset.
- **IMAGE_DIR**: Directory containing the product images.
- **OUTPUT_CSV_PATH**: File path for saving the predictions.
- **BATCH_SIZE**: Batch size for data loading.
- **NUM_WORKERS**: Number of workers for data loading.
- **LEARNING_RATE**: Learning rate for the optimizer.
- **NUM_EPOCHS**: Number of epochs for training.
- **DATASET_SIZE**: Maximum number of samples to load from the dataset.

## Logging

The code uses Python’s `logging` module to log important information such as the device being used (GPU/CPU), training progress, and errors encountered while loading images or during training.

## Error Handling

If an image fails to load, the program logs an error and continues by using a placeholder tensor for that image. This ensures that the training process is not interrupted due to file issues.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- PIL (Pillow)
- tqdm
- scikit-learn