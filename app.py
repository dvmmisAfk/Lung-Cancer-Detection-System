import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
from flask import Flask, request, render_template, jsonify
import io

app = Flask(__name__)

# Configuration
CLASS_NAMES = ['Bengin cases', 'Malignant cases', 'Normal cases']
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TOP_K_FEATURES = 4323
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden1=100, hidden2=70, n_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(hidden2, n_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Load models
def load_models():
    # Use relative paths instead of absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "mlp_model.pth")
    indices_path = os.path.join(base_dir, "models", "top_k_indices.npy")
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"mlp_model.pth not found at {model_path}. Please place the model file in the models directory.")
    if not os.path.exists(indices_path):
        raise FileNotFoundError(f"top_k_indices.npy not found at {indices_path}. Please place the indices file in the models directory.")

    # Load VGG16
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    for param in vgg.parameters():
        param.requires_grad = False
    vgg.to(DEVICE)
    vgg.eval()

    # Load SimpleMLP
    mlp_model = SimpleMLP(TOP_K_FEATURES, hidden1=100, hidden2=70, n_classes=3)
    mlp_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    mlp_model.to(DEVICE)
    mlp_model.eval()

    # Load indices
    top_k_indices = np.load(indices_path)

    return vgg, mlp_model, top_k_indices

# CT scan check
def is_ct_scan(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    return (50 < mean_intensity < 200) and (std_intensity > 20)

# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform(image).unsqueeze(0)

# Extract features
def extract_features(vgg_model, image_tensor):
    with torch.no_grad():
        features = vgg_model.features(image_tensor)
        features = vgg_model.avgpool(features)
        features = torch.flatten(features, 1)
    return features.cpu().numpy()

# Initialize models as None
vgg_model, mlp_model, top_k_indices = None, None, None
MODELS_LOADED = False

# Try to load models when the app starts
try:
    vgg_model, mlp_model, top_k_indices = load_models()
    MODELS_LOADED = True
    print("Models loaded successfully.")
except Exception as e:
    MODELS_LOADED = False
    print(f"Model loading failed: {e}")
    print("Please make sure the model files (mlp_model.pth and top_k_indices.npy) are in the 'models' directory.")

@app.route('/')
def index():
    return render_template('index.html', models_loaded=MODELS_LOADED)

@app.route('/predict', methods=['POST'])
def predict():
    if not MODELS_LOADED:
        return jsonify({'error': 'Models not loaded. Please check server configuration.'})
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        if not is_ct_scan(image):
            return jsonify({
                'is_ct_scan': False,
                'message': 'Uploaded image does not appear to be a CT scan.'
            })
        
        image_tensor = preprocess_image(image).to(DEVICE)
        features = extract_features(vgg_model, image_tensor)
        features_selected = features[:, top_k_indices]
        features_tensor = torch.from_numpy(features_selected).float().to(DEVICE)

        with torch.no_grad():
            output = mlp_model(features_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        predicted_label = CLASS_NAMES[predicted_class.item()]
        confidence_percent = confidence.item() * 100

        return jsonify({
            'is_ct_scan': True,
            'prediction': predicted_label,
            'confidence': f"{confidence_percent:.2f}%",
            'message': f'The CT scan appears to show: {predicted_label} (Confidence: {confidence_percent:.2f}%)'
        })

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'})

@app.route('/health')
def health_check():
    return jsonify({
        'models_loaded': MODELS_LOADED,
        'device': str(DEVICE)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)