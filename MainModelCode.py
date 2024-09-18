import os
import csv
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CSV_PATH = r"C:\Users\SALIL HIREMATH\Downloads\train.csv"
IMAGE_DIR = r"C:\Users\SALIL HIREMATH\Downloads\images"
OUTPUT_CSV_PATH = r"C:\Users\SALIL HIREMATH\Downloads\predictions.csv"
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
DATASET_SIZE = 80000

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("GPU not available. Using CPU.")

# Custom Dataset
class ProductImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, group_id, entity_name, entity_value = self.data[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, group_id, entity_name, entity_value
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return torch.zeros((3, 224, 224)), group_id, entity_name, entity_value

# Model Architecture
class ProductAttributeModel(nn.Module):
    def __init__(self, num_group_ids, num_entity_names):
        super(ProductAttributeModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fc1 = nn.Linear(num_ftrs, 512)
        self.fc2 = nn.Linear(512, num_group_ids)
        self.fc3 = nn.Linear(512, num_entity_names)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        group_id = self.fc2(x)
        entity_name = self.fc3(x)
        return group_id, entity_name

# Load and preprocess dataset
def load_dataset(csv_path, limit=None):
    data = []
    group_ids = set()
    entity_names = set()
    
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            image_link, group_id, entity_name, entity_value = row
            image_filename = os.path.basename(image_link)
            image_path = os.path.join(IMAGE_DIR, image_filename)
            
            if os.path.exists(image_path):
                data.append((image_path, group_id, entity_name, entity_value))
                group_ids.add(group_id)
                entity_names.add(entity_name)
    
    return data, list(group_ids), list(entity_names)

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS):
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        logger.info('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects_group = 0
            running_corrects_entity = 0

            for inputs, group_ids, entity_names, _ in tqdm(dataloader, desc=phase):
                inputs = inputs.to(device)
                group_ids = group_ids.to(device)
                entity_names = entity_names.to(device)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    with torch.set_grad_enabled(phase == 'train'):
                        group_outputs, entity_outputs = model(inputs)
                        loss_group = criterion(group_outputs, group_ids)
                        loss_entity = criterion(entity_outputs, entity_names)
                        loss = loss_group + loss_entity

                        if phase == 'train':
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects_group += torch.sum(torch.argmax(group_outputs, 1) == group_ids)
                running_corrects_entity += torch.sum(torch.argmax(entity_outputs, 1) == entity_names)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc_group = running_corrects_group.double() / len(dataloader.dataset)
            epoch_acc_entity = running_corrects_entity.double() / len(dataloader.dataset)

            logger.info(f'{phase} Loss: {epoch_loss:.4f} Group Acc: {epoch_acc_group:.4f} Entity Acc: {epoch_acc_entity:.4f}')

    return model

# Prediction Function
def predict_and_save(model, test_loader, group_ids, entity_names, output_csv_path):
    model.eval()

    predictions = []

    with torch.no_grad():
        for inputs, _, _, entity_values in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device)
            group_outputs, entity_outputs = model(inputs)
            
            pred_group_ids = [group_ids[i] for i in torch.argmax(group_outputs, 1).cpu().numpy()]
            pred_entity_names = [entity_names[i] for i in torch.argmax(entity_outputs, 1).cpu().numpy()]
            
            for group_id, entity_name, entity_value in zip(pred_group_ids, pred_entity_names, entity_values):
                predictions.append([group_id, entity_name, entity_value])

    # Save predictions to CSV
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['predicted_group_id', 'predicted_entity_name', 'entity_value'])
        writer.writerows(predictions)

    logger.info(f"Predictions saved to {output_csv_path}")

# Main execution
def main():
    try:
        # Load dataset
        data, group_ids, entity_names = load_dataset(CSV_PATH, limit=DATASET_SIZE)
        
        if not data:
            logger.error("No valid data was found. Please check your CSV file and image downloads.")
            return

        # Create label mappings
        group_id_to_idx = {gid: idx for idx, gid in enumerate(group_ids)}
        entity_name_to_idx = {name: idx for idx, name in enumerate(entity_names)}

        # Prepare data for PyTorch
        labeled_data = [
            (
                image_path,
                group_id_to_idx[group_id],
                entity_name_to_idx[entity_name],
                entity_value
            )
            for image_path, group_id, entity_name, entity_value in data
        ]

        # Split dataset
        train_data, val_data = train_test_split(labeled_data, test_size=0.2, random_state=42)
        
        # Create datasets and dataloaders
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        train_dataset = ProductImageDataset(train_data, transform=transform)
        val_dataset = ProductImageDataset(val_data, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        
        # Initialize model, loss function, and optimizer
        model = ProductAttributeModel(len(group_ids), len(entity_names))
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Train model
        trained_model = train_model(model, train_loader, val_loader, criterion, optimizer)
        
        # Make predictions and save to CSV
        predict_and_save(trained_model, val_loader, group_ids, entity_names, OUTPUT_CSV_PATH)
        
        # Save the model
        torch.save(trained_model.state_dict(), 'product_attribute_model.pth')
        logger.info("Model saved as 'product_attribute_model.pth'")

    except Exception as e:
        logger.error(f"An error occurred in the main execution: {str(e)}")

if __name__ == "__main__":
    main()