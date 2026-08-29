# Interpreting Deep CNN Decisions in Multi-class Brain Tumor Classification via Grad-CAM

This repository contains the codebase and methodology for our research on multi-class brain tumor classification using Magnetic Resonance Imaging (MRI)[cite: 5]. The project addresses the "black box" problem in deep learning by integrating EfficientNetB1 with Gradient-weighted Class Activation Mapping (Grad-CAM) to provide a transparent, trustworthy decision-support tool for clinicians[cite: 5].

## Dataset
The project utilizes the **Brain Tumor MRI Dataset** by Masoud Nickparvar, available on Kaggle[cite: 5].
*   **Total Images:** 7,200 MRI scans[cite: 5]
*   **Classes:** Glioma, Meningioma, Pituitary Tumor, No Tumor[cite: 5]
*   **Data Split:** 4,480 for Training (80% of training subset), 1,120 for Validation (20% of training subset), and a strictly held-out set of 1,600 for Testing[cite: 5].

## Preprocessing Pipeline
To maximize model performance and isolate the brain region from surrounding tissues, an OpenCV-based skull-stripping pipeline was implemented[cite: 5]:
*   Grayscale conversion and 5x5 Gaussian blur to minimize high-frequency noise[cite: 5].
*   Binary thresholding applied at a fixed intensity of 45 using `cv2.THRESH_BINARY`[cite: 5].
*   Two iterations of dilation to recover reduced brain structures[cite: 5].
*   Extreme point bounding box cropping using `cv2.findContours` with the `cv2.RETR_EXTERNAL` flag[cite: 5].
*   Images resized to 240x240 pixels[cite: 5].
*   Data augmentation (random rotation, vertical shifting, horizontal flipping) applied strictly to the training subset to prevent data leakage[cite: 5].

## Model Architectures
Three Convolutional Neural Network (CNN) architectures were evaluated for this multi-class problem[cite: 5]:
1.  **EfficientNetB1 (Proposed Model):** Fine-tuned on the last 50 layers with a custom classification head (dimensionality reduction, 256-unit dense layer, dropout layers, and 4-unit softmax)[cite: 5].
2.  **VGG16:** Used as a transfer learning comparison model[cite: 5].
3.  **Custom CNN:** Used as a baseline model to demonstrate the limitations of standard architectures[cite: 5].

## Results
EfficientNetB1 emerged as the most robust architecture, demonstrating stable convergence and minimal cross-class confusion (e.g., distinguishing between Glioma and Meningioma)[cite: 5]. 

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Custom CNN** | 81.75% | 83.19% | 81.75% | 81.60% |
| **VGG16** | 94.81% | 95.05% | 94.81% | 94.70% |
| **EfficientNetB1** | 95.19% | 95.00% | 95.00% | 95.00% |

## Explainable AI (Grad-CAM)
To ensure clinical applicability, Grad-CAM was integrated to visualize the specific anatomical regions driving the network's predictions[cite: 5]. The generated heatmaps successfully highlight the most influential features, providing precise spatial localization of pathological regions and ensuring the model acts as a transparent second opinion rather than an autonomous black box[cite: 5].

## Installation & Usage
1. Clone this repository: `git clone https://github.com/yourusername/your-repo-name.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset from Kaggle and place it in the `data/` directory.
4. Run the preprocessing script: `python preprocess.py`
5. Train the models: `python train.py`
6. Generate Grad-CAM heatmaps: `python evaluate_gradcam.py`

## Team & Authors
*   **Michelle Pricillia Sutanto** (School of Computer Science, Binus University)[cite: 5]
*   **Aaron Faustine Suryanto** (School of Computer Science, Binus University)[cite: 5]
*   **Beni Mulayawan** (School of Computer Science, Binus University)[cite: 5]
*   **Jurike V. Moniaga, S.Kom., M.T.** (School of Computer Science, Binus University)[cite: 5]
*   **Setiawan Joddy, S.Kom., M.Kom.** (School of Computer Science, Binus University)[cite: 5]

*Submitted to the 2026 International Conference on Information Management and Technology (ICIMTech).*
