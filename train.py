import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from model import CustomCNN  

DATASET_PATH = 'inaturalist_12K/train'  
SAMPLES_PER_CLASS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


dataset = ImageFolder(root=DATASET_PATH, transform=transform)
class_names = dataset.classes

indices = []
class_counter = {cls_idx: 0 for cls_idx in range(len(class_names))}

for idx, (_, label) in enumerate(dataset):
    if class_counter[label] < SAMPLES_PER_CLASS:
        indices.append(idx)
        class_counter[label] += 1
    if all(v >= SAMPLES_PER_CLASS for v in class_counter.values()):
        break

subset = Subset(dataset, indices)
dataloader = DataLoader(subset, batch_size=8, shuffle=False)

model = CustomCNN(
    channels=3,
    num_filters=64,
    kernel_size=3,
    activation_fn=torch.nn.ReLU,
    dense_neurons=256,
    num_classes=len(class_names)
)

model = model.to(DEVICE)
model.eval()

def show_predictions(model, dataloader, class_names, output_dir="inference_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        image_count = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for i in range(images.size(0)):
                img = images[i].cpu().permute(1, 2, 0).numpy()
                true_label = class_names[labels[i].item()]
                predicted_label = class_names[preds[i].item()]

                plt.imshow(img)
                plt.title(f"Pred: {predicted_label} | Actual: {true_label}")
                plt.axis('off')

                save_path = os.path.join(
                    output_dir,
                    f"img_{image_count:03d}_pred_{predicted_label}_actual_{true_label}.png"
                )
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
                image_count += 1

if __name__ == "__main__":
    show_predictions(model, dataloader, class_names)
