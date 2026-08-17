# DeepFake Detection System Report

## 1. Dataset Description
The dataset was formulated using the `FakeAVCeleb` raw videos. These were dynamically split and strictly categorized by identity into Train (70%), Evaluation (15%), and Test (15%) splits avoiding identity-leakage during validation. From the video pipeline, each instance was extracted into spatial frame imagery (`.jpg`).

## 2. Model Architecture
A PyTorch Transfer Learning architecture was employed.
- **Backbone**: Torchvision `ResNet18`
- **Pretrained**: Yes (ImageNet `DEFAULT` Weights)
- **Classifier Head**: Replaced the fully connected layer with `Linear(512) -> ReLU -> Dropout(0.3) -> Linear(1)`
The model outputs a raw logit, representing binary classification (0 = Real, 1 = Fake) when mapped via Sigmoid.

## 3. Training Setup
- **Data Augmentation**: `RandomCrop(224)` nested out of a `256` resize, alongside `RandomHorizontalFlip()` and `ColorJitter()` to ensure invariance to lighting and translation.
- **Loss**: `BCEWithLogitsLoss()`
- **Optimizer**: `Adam` (Learning Rate = `1e-4`)
- **Hardware Profile**: Automatically defaulted to dynamic GPU `cuda:0`

## 4. Evaluation 
The model demonstrates an robust extraction mechanism on the hidden Test set. 
By saving graphical analyses such as the Confusion Matrix (`outputs/plots/confusion_matrix.png`) and plotting the ROC Area Under the Curve (`outputs/plots/roc_curve.png`), the false positive rate handles reliably against artifacts.

## 5. System Features
The system now functions cleanly handling:
- Multi-Frame Inference Averaging
- Misclassification Logging (Saves visually to `outputs/misclassified/`)
- Prediction Visualization Script
- Extensible API logic 

## 6. Observations 
Implementing `RandomCrop` and Color augmentations demonstrably improved the system's resilience against color-grading disparities found across differently compressed fake generated data. Incorporating Multi-Frame predictions stabilizes jittery per-frame confidences found commonly in deepfake videos.
