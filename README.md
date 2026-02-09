# 🎗️ Breast Cancer Detection (IDC Classification)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)

This project implements a Deep Learning model to classify breast cancer histopathology images. It specifically targets **Invasive Ductal Carcinoma (IDC)** using a custom Convolutional Neural Network (CNN).

---

## 📊 Project Overview

- **Goal:** Detecting IDC in breast tissue patches.
- **Dataset:** Breast Histopathology Images (Kaggle).
- **Model:** Sequential CNN built with TensorFlow/Keras.
- **Accuracy:** ~84.89% on the test set.

## 📂 Project Structure

- `BCD.ipynb`: Data downloading, preprocessing, and model training.
- `app.py`: Streamlit web application for real-time inference.
- `model.json` & `model.weights.h5`: Saved model architecture and weights.

## 🚀 Installation & Usage

### 1. Clone the Repo

```bash
git clone [https://github.com/AlirezaSaberian/BreastCancerDetection_Python.git](https://github.com/AlirezaSaberian/BreastCancerDetection_Python.git)
cd BreastCancerDetection_Python
```

### 2. Activate Virtual Environment

```bash
source tf-venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

Frameworks: TensorFlow, Keras

Interface: Streamlit

Data Science: NumPy, Pandas, Scikit-learn, Matplotlib

Automation: Kagglehub (for automated data download)

## ⚠️ Important Note

The large dataset folders (archive/, all_rays_dir/, etc.) and the virtual environment (tf-venv/) are not uploaded to this repository to save space. The notebook will automatically download the data via kagglehub when executed.
