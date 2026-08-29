# Brain Tumor Classification & XAI 🧠

A deep learning project focused on interpreting and explaining deep CNN decisions in multi-class brain tumor classification utilizing Grad-CAM and SHAP. Built as an advanced Computer Vision academic project by Computer Science students at BINUS University. 🚀

*(Disclaimer: This project is designed for educational and AI research purposes and is not intended to be a substitute for professional medical diagnosis or advice).*

## Features ✨

* **Multi-Class Tumor Classification:** Accurately classifies MRI scans into different categories of brain tumors using a robust Deep Convolutional Neural Network (CNN) architecture.
* **Explainable AI (XAI) Integration:** Demystifies the "black box" of deep learning by providing visual explanations of the model's decision-making process.
* **Grad-CAM Visualization:** Generates heatmaps highlighting the specific regions of the MRI scan that most heavily influenced the model's prediction.
* **SHAP Value Analysis:** Utilizes Shapley Additive exPlanations to interpret the impact of individual features and pixels on the final classification output.

## Tech Stack 💻

* **Machine Learning & Deep Learning:** PyTorch / TensorFlow, Keras, Scikit-learn
* **Explainable AI (XAI):** Grad-CAM, SHAP
* **Data Processing & Vision:** OpenCV, NumPy, Pandas, Pillow
* **Data Visualization:** Matplotlib, Seaborn
* **Language:** Python 3

## Project Structure 📂

```text
📦 Brain_Tumor
 ┣ 📂 dataset           # Directory for brain MRI image datasets (Not included in repo due to size)
 ┣ 📂 models            # Saved Deep CNN model weights (.h5 / .pth)
 ┣ 📂 notebooks         # Jupyter notebooks for EDA, training, and XAI experiments
 ┣ 📂 src               
 ┃ ┣ 📜 model.py        # CNN architecture definition
 ┃ ┣ 📜 train.py        # Training and evaluation script
 ┃ ┣ 📜 explainers.py   # Grad-CAM and SHAP implementation logic
 ┃ ┗ 📜 utils.py        # Image preprocessing and helper functions
 ┣ 📜 requirements.txt  # Python Dependencies
 ┗ 📜 README.md         # Project Documentation
