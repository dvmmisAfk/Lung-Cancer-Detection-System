Lung Cancer Detection System

A deep learning–based lung cancer detection system with a Flask backend.
This model classifies CT scan images of lungs into three categories:

→ Benign cases
→ Malignant cases
→ Normal cases

The system is designed to reject non-CT scan images to ensure robustness against irrelevant inputs.
The trained model is served via a Flask API (`app.py`), enabling web or mobile integration.

interface-
<img width="674" height="505" alt="image" src="https://github.com/user-attachments/assets/fe8c8e2a-b818-454b-a7d8-bb7e0bed297d" />

---

## Features

→ Rejects non-CT scan images automatically
→ Uses VGG16 pretrained on ImageNet for feature extraction
→ Feature importance and selection using ExtraTreesClassifier
→ Lightweight MLP classifier trained on top-ranked features
→ Evaluation with accuracy, precision, recall, F1-score, and confusion matrix
→ Visualization of training history, feature importance, and predictions
→ Flask API (`app.py`) for easy deployment and integration

---

## Dataset Structure

dataset/Lung/
    ├── Benign cases/
    ├── Malignant cases/
    └── Normal cases/

---

## Model Architecture

1. Preprocessing
   → Resize images to 224×224
   → Normalize with ImageNet mean & std
   → Reject non-CT scan inputs

2. Feature Extraction
   → Pretrained VGG16 (feature layers frozen)
   → Extracts deep features from CT images

3. Feature Selection
   → ExtraTreesClassifier ranks features
   → Top 4323 features are retained

4. Classifier (MLP)
   → Input → Hidden (100 neurons, ReLU) → Hidden (70 neurons, ReLU) → Output (3 classes)
   → Loss: CrossEntropyLoss
   → Optimizer: Adam (lr=1e-3)
   → Epochs: 15, Batch Size: 32

---

## Training and Evaluation

→ Accuracy tracked across epochs
→ Confusion matrix for test predictions
→ Classification report: Precision, Recall, F1-score
→ Training plots: Loss & Accuracy over epochs

Example evaluation flow:

acc, preds, labels = evaluate_mlp(model, test_loader, device)
print("Test Accuracy:", acc)
print(classification_report(labels, preds, target_names=class_names))
plot_confusion_matrix(labels, preds, class_names)

---

## Inference (Single Image Prediction)

Example for classifying a new CT scan image:

prediction, confidence, image = classify_image(
    "sample_ct_scan.png",
    vgg_model,
    mlp_model,
    top_k_indices,
    class_names,
    device
)

print(f"Prediction: {prediction}, Confidence: {confidence:.2%}")
display_image_with_prediction(image, prediction, confidence)

---

## Flask Backend (`app.py`)

The project includes a Flask backend to serve predictions as an API.

Run the Flask app:

python app.py


Default address:
[http://127.0.0.1:5000/](http://127.0.0.1:5000/)

Example API request:

curl -X POST -F "file=@sample_ct_scan.png" http://127.0.0.1:5000/predict


Example JSON response:

{
  "prediction": "Malignant",
  "confidence": 0.92
}

---

## Installation

Install dependencies:

pip install numpy torch torchvision torchaudio scikit-learn tqdm matplotlib seaborn opencv-python flask

---

## Usage

1. Clone this repository

2. Prepare dataset in `dataset/Lung/`

3. Train and evaluate the model:

   python train.py

4. Run Flask backend:

   python app.py


5. Send image requests to the API for predictions

---

## Results

→ Multi-class classification on Benign, Malignant, and Normal cases
→ Visualizations include:

* Training loss & accuracy over epochs
* Confusion matrix heatmap
* Feature importance plots
* Prediction results with confidence

(Insert final accuracy, F1-score, and confusion matrix plot after training)

---

## Tech Stack

→ PyTorch – Deep learning framework
→ Torchvision – Pretrained models & transforms
→ Scikit-learn – Feature selection & evaluation metrics
→ OpenCV + PIL – Image preprocessing
→ Matplotlib + Seaborn – Visualization
→ Flask – Backend API service

---

## License

This project is for academic and research purposes only.
Not intended for clinical or commercial medical use.
