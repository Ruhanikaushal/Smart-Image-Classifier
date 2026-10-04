import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# 1. Define model architecture matching training
class PlantDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Identity()
        
        self.fc1 = nn.Linear(in_features, 512)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load classes
    classes_path = 'classes.json' if os.path.exists('classes.json') else os.path.join('..', 'classes.json')
    with open(classes_path, 'r') as f:
        class_names = json.load(f)

    # Initialize and load weights
    model_path = 'best_plant_model.pth' if os.path.exists('best_plant_model.pth') else os.path.join('..', 'best_plant_model.pth')
    model = PlantDiseaseClassifier(num_classes=len(class_names))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dir = 'test_images' if os.path.exists('test_images') else os.path.join('..', 'test_images')
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\nFound {len(image_files)} test images in '{test_dir}':")

    results = []
    for img_name in sorted(image_files):
        img_path = os.path.join(test_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            top_probs, top_indices = torch.topk(probs, 3)

        top_1_class = class_names[top_indices[0].item()]
        top_1_conf = top_probs[0].item() * 100

        print(f"\nImage: {img_name}")
        print(f"  Top Prediction: {top_1_class} ({top_1_conf:.2f}%)")
        print("  Top 3 Candidates:")
        for rank in range(3):
            cls_name = class_names[top_indices[rank].item()]
            conf = top_probs[rank].item() * 100
            print(f"    {rank+1}. {cls_name}: {conf:.2f}%")

        results.append({
            'filename': img_name,
            'image': img,
            'top_class': top_1_class,
            'confidence': top_1_conf,
            'top_3': [(class_names[top_indices[r].item()], top_probs[r].item()*100) for r in range(3)]
        })

    # Visualize test results grid
    n = len(results)
    if n > 0:
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, res in zip(axes, results):
            ax.imshow(res['image'])
            formatted_title = res['top_class'].replace('___', '\n').replace('_', ' ')
            ax.set_title(f"{formatted_title}\nConf: {res['confidence']:.1f}%", fontsize=10, fontweight='bold', color='darkgreen')
            ax.axis('off')
        plt.tight_layout()
        out_grid = os.path.join('Documents', 'test_predictions_grid.png') if os.path.exists('Documents') else 'test_predictions_grid.png'
        plt.savefig(out_grid, dpi=300)
        plt.close()
        print(f"\nSaved visualization to '{out_grid}'")

if __name__ == '__main__':
    main()
