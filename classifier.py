import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

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

class InferenceEngine:
    def __init__(self, model_path='best_plant_model.pth', classes_path='classes.json'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(classes_path, 'r') as f:
            self.class_names = json.load(f)
            
        self.model = PlantDiseaseClassifier(num_classes=len(self.class_names))
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image: Image.Image, top_k=5):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            top_probs, top_indices = torch.topk(probs, min(top_k, len(self.class_names)))
            
        results = []
        for i in range(len(top_indices)):
            cls_idx = top_indices[i].item()
            cls_name = self.class_names[cls_idx]
            conf = float(top_probs[i].item())
            
            # Parse crop and disease name
            parts = cls_name.split('___')
            crop = parts[0].replace('_', ' ')
            condition = parts[1].replace('_', ' ') if len(parts) > 1 else 'healthy'
            
            results.append({
                'raw_class': cls_name,
                'crop': crop,
                'condition': condition,
                'confidence': conf,
                'confidence_pct': conf * 100
            })
        return results
