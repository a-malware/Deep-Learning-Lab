import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import SimpleSpeakerCNN

# Configuration
PROCESSED_DIR = "data/processed_clean"  # Use clean spectrograms
MODEL_SAVE_PATH = "models/speaker_cnn_final.pth"
BATCH_SIZE = 16
LEARNING_RATE = 0.0003
NUM_EPOCHS = 150
IMG_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpectrogramDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_dataset():
    """Load all spectrogram images and create labels"""
    image_paths = []
    labels = []
    speaker_to_idx = {}
    
    speakers = sorted([d for d in os.listdir(PROCESSED_DIR) 
                      if os.path.isdir(os.path.join(PROCESSED_DIR, d))])
    
    print(f"Found {len(speakers)} speakers: {speakers}")
    
    for idx, speaker in enumerate(speakers):
        speaker_to_idx[speaker] = idx
        speaker_dir = os.path.join(PROCESSED_DIR, speaker)
        
        for filename in os.listdir(speaker_dir):
            if filename.endswith('.png'):
                image_paths.append(os.path.join(speaker_dir, filename))
                labels.append(idx)
    
    print(f"Total images: {len(image_paths)}")
    print(f"Speaker mapping: {speaker_to_idx}")
    
    return image_paths, labels, speaker_to_idx

def create_data_loaders(image_paths, labels):
    """Split data and create data loaders"""
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Split: 85% train, 15% test
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    print(f"\nDataset split:")
    print(f"Train: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    
    train_dataset = SpectrogramDataset(X_train, y_train, transform=transform)
    test_dataset = SpectrogramDataset(X_test, y_test, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def test_model(model, test_loader, device, speaker_to_idx):
    """Test the model and generate metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    idx_to_speaker = {v: k for k, v in speaker_to_idx.items()}
    target_names = [idx_to_speaker[i] for i in sorted(idx_to_speaker.keys())]
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Test Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Final Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_final.png')
    print("\nConfusion matrix saved to 'confusion_matrix_final.png'")
    
    return accuracy

def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss', marker='o', color='blue', alpha=0.7)
    ax1.plot(test_losses, label='Test Loss', marker='s', color='red', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_accs, label='Train Accuracy', marker='o', color='green', alpha=0.7)
    ax2.plot(test_accs, label='Test Accuracy', marker='s', color='orange', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history_final.png')
    print("Training history saved to 'training_history_final.png'")

def main():
    print("="*60)
    print("FINAL MODEL TRAINING - CLEAN DATASET")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Data directory: {PROCESSED_DIR}")
    print("="*60)
    
    if not os.path.exists(PROCESSED_DIR):
        print(f"\nERROR: Clean data directory '{PROCESSED_DIR}' not found!")
        print("Please run: python preprocess_clean.py")
        return
    
    image_paths, labels, speaker_to_idx = load_dataset()
    num_speakers = len(speaker_to_idx)
    
    train_loader, test_loader = create_data_loaders(image_paths, labels)
    
    model = SimpleSpeakerCNN(num_speakers=num_speakers).to(DEVICE)
    print(f"\nModel initialized with {num_speakers} output classes")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                      factor=0.5, patience=15)
    
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    best_test_acc = 0.0
    patience_counter = 0
    early_stop_patience = 25
    
    print("\n" + "="*60)
    print("TRAINING STARTED")
    print("="*60)
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
        
        scheduler.step(test_acc)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            os.makedirs("models", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'speaker_to_idx': speaker_to_idx,
                'model_type': 'simple'
            }, MODEL_SAVE_PATH)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  → Best model saved! (Test Acc: {test_acc:.2f}%)")
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_accuracy = test_model(model, test_loader, DEVICE, speaker_to_idx)
    
    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
