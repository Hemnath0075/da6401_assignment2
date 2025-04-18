import torch
import torch.nn as nn
import logging
from datetime import datetime

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f'logs/image_size_300_scale_0.5_to_1.0_{timestamp}.txt'

# Set up logging with the timestamped filename
logging.basicConfig(filename=filename, level=logging.INFO, format='%(asctime)s - %(message)s')


class SmallCNN(nn.Module):
    def __init__(self, config):
        super(SmallCNN, self).__init__()

        in_channels = 3  # RGB
        conv_layers = []

        for i in range(5):
            conv_layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=config['conv_filters'][i],
                kernel_size=config['kernel_sizes'][i],
                padding=1
            ))
            conv_layers.append(config['conv_activation']())
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = config['conv_filters'][i]

        self.conv_block = nn.Sequential(*conv_layers)

        dummy_input = torch.zeros(1, 3, *config['image_size'])
        dummy_output = self.conv_block(dummy_input)
        flat_size = dummy_output.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat_size, config['dense_neurons'])
        self.act_dense = config['dense_activation']()
        self.output = nn.Linear(config['dense_neurons'], config['num_classes'])

    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x, 1)
        x = self.act_dense(self.fc1(x))
        return self.output(x)
    


import torch
import torch.nn as nn
import torch.optim as optim

class TrainerPredictor:
    def __init__(self, model, device, class_names, lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, train_loader, val_loader, epochs=10):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            val_acc = self.validate(val_loader)
            logging.info(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Val Acc: {val_acc:.2f}%")

    def validate(self, val_loader):
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return 100 * correct / total

    def predict(self, image_tensor):
        self.model.eval()
        image_tensor = image_tensor.to(self.device).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, pred = torch.max(outputs, 1)
        
        return self.class_names[pred.item()]
    
    
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


# Configuration
config = {
    'conv_filters': [32, 64, 128, 256, 256],
    'kernel_sizes': [3, 3, 3, 3, 3],
    'conv_activation': nn.ReLU,
    'dense_activation': nn.ReLU,
    'dense_neurons': 512,
    'num_classes': 10,
    'image_size': (300, 300)
}

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(config['image_size'], scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
])


# For validation, keep only resizing and tensor conversion (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize(config['image_size']),
    transforms.ToTensor(),
])


train_dataset = datasets.ImageFolder('train', transform=train_transform)
val_dataset = datasets.ImageFolder('val', transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize
model = SmallCNN(config)
trainer = TrainerPredictor(model, device, train_dataset.classes)

# Train
trainer.train(train_loader, val_loader, epochs=10)



