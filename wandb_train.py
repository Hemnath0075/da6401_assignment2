import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.optim import Adam

import sys
import os
from datetime import datetime

# Create a logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Timestamped log file
log_filename = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Redirect stdout and stderr to the log file
sys.stdout = open(log_filename, 'w')
sys.stderr = sys.stdout


activations = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'leaky_relu': nn.LeakyReLU(),
    'mish':nn.Mish(),
    'gelu':nn.GELU(),
    'silu':nn.SiLU(),
    'relu6':nn.ReLU6()
}

optimizer_dict = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'sgd': optim.SGD
}


def generate_filters(base_m, strategy):
            if strategy == 'same':
                return [base_m] * 5
            elif strategy == 'double':
                return [base_m * (2 ** i) for i in range(5)]
            elif strategy == 'half':
                return [max(1, base_m // (2 ** i)) for i in range(5)]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        
        in_channels = config['input_dimension'][0]
        base_m = config['conv_filters']
        strategy = config['filter_org']
        conv_filters = generate_filters(base_m, strategy)
        kernel_sizes = config['kernel_sizes']
        stride = config['stride']
        padding = config['padding']
        pool = config['max_pooling_size']
        dropout = config['dropout_rate']
        use_bn = config['use_batchnorm']
        dropout_org = config['dropout_organisation']

        conv_layers = []
        for i in range(5):  # 5 conv layers
            out_channels = conv_filters[i]
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[i], stride=stride, padding=padding))
            if use_bn:
                conv_layers.append(nn.BatchNorm2d(out_channels))
            if dropout_org == 'before_relu':
                conv_layers.append(nn.Dropout2d(dropout))
            conv_layers.append(activations[config['conv_activation']])
            if dropout_org == 'after_relu':
                conv_layers.append(nn.Dropout2d(dropout))
            conv_layers.append(nn.MaxPool2d(kernel_size=pool))
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        # Estimate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros((1, *config['input_dimension']))
            dummy_output = self.conv(dummy_input)
            flattened_size = dummy_output.view(1, -1).shape[1]

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, config['dense_neurons']),
            activations[config['dense_activation']],
            nn.Dropout(dropout),
            nn.Linear(config['dense_neurons'], config['num_classes'])
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TrainAndPredict:
    def __init__(self, model, device, class_names, optimizer=None, lr=0.001, weight_decay=0.0):
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer_dict[optimizer](self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, train_loader, val_loader, epochs=10, save_path='best_model.pth'):
        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct, total = 0, 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = total_loss / len(train_loader)
            train_acc = 100 * correct / total
            val_acc = self.validate(val_loader)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")


            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), save_path)

                artifact = wandb.Artifact('best-model', type='model')
                artifact.add_file(save_path)
                wandb.log_artifact(artifact)

            # Log to Weights & Biases
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            })

        print(f"\nBest model saved from Epoch {best_epoch} with Val Acc: {best_val_acc:.2f}%")

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



def train_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config
        # print(config.conv_filters)
        wandb.run.name = f"filter_{config.filter_size}/dn_{config.n_neurons}/opt_{config.optimizer}/aug_{config.use_augmentation}"

        # Build dynamic config from sweep values
        dynamic_config = {
            'input_dimension': (3, 224, 224),
            'conv_filters': config.conv_filters,
            'kernel_sizes': [config.filter_size] * 5,
            'stride': config.stride,
            'filter_org': config.filter_org,
            'padding': config.padding,
            'max_pooling_size': config.max_pooling_size,
            'dropout_rate': config.dropout_rate,
            'use_batchnorm': config.use_batchnorm,
            'factor': config.factor,
            'dropout_organisation': 'after_relu',
            'dense_neurons': config.n_neurons,
            'num_classes': config.n_classes,
            'optimizer': config.optimizer,
            'conv_activation': config.conv_activation,
            'dense_activation': config.dense_activation,
            'image_size': (224, 224),
            
        }
        
        if config['filter_org'] == 'half' and config['conv_filters'] < 32:
            print("Skipping config: unsafe filter_org with too few filters")
            return
        if config['stride'] > 1 and config['max_pooling_size'] > 1 and config['filter_size'] >= 7:
            print("Skipping config: stride/pool too aggressive with large filter")
            return

        # Define your model
        model = CNN(dynamic_config)

        # Dataloaders
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(dynamic_config['image_size'], scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
        ]) if config.use_augmentation else transforms.Compose([
            transforms.Resize(dynamic_config['image_size']),
            transforms.ToTensor(),
        ])

        val_transform = transforms.Compose([
            transforms.Resize(dynamic_config['image_size']),
            transforms.ToTensor(),
        ])
        
        
        

        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
        train_dataset = datasets.ImageFolder('train', transform=train_transform)
        val_dataset = datasets.ImageFolder('val', transform=val_transform)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True,num_workers=4, pin_memory=True)

        trainer = TrainAndPredict(model, device, train_dataset.classes,optimizer=config.optimizer,lr=config.learning_rate)

        # Train and log
        trainer.train(train_loader, val_loader, epochs=config.epochs)
        

sweep_config = {
    'method': 'bayes',
    'name': 'Custom CNN',
    'metric': {'name': "val_accuracy", 'goal': 'maximize'},
    'parameters': {
        'conv_filters': {'values': [32, 64, 128]},
        'filter_org': {
            'values': ['same', 'double', 'half']
        },
        'filter_size': {'values': [1,3,7,11]},
        'stride': {'values': [1,2]},
        'padding': {'values': [1,2]},
        'max_pooling_size': {'value': 2},
        'n_neurons': {'values': [64, 128, 256, 512, 1024]},
        'n_classes': {'value': 10},
        'conv_activation': {
            'values': ['relu', 'gelu', 'silu', 'mish', 'relu6','leaky_relu']
        },
        'dense_activation': {
            'values': ['relu', 'gelu', 'silu', 'mish', 'relu6','leaky_relu']
        },
        'dropout_rate': {'values': [0.2, 0.3, 0.4, 0.5]},
        'use_batchnorm': {'values': [True, False]},
        'factor': {'values': [0.5, 1, 2, 3]},
        'learning_rate': {'values': [0.001,0.0001]},
        'batch_size': {'values': [16,32,64]},
        'optimizer': {'values': ['adam', 'adamw','sgd']},
        'epochs': {'values': [5,10,15]},
        'use_augmentation': {'values': [True, False]},
        'dropout_organisation': {'values': ['after_relu','before_relu']},  # simplified for now
    },
}

sweep_id = wandb.sweep(sweep_config, project="iNaturalist_CNN")
wandb.agent(sweep_id, function=train_sweep, count=30)
