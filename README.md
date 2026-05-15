# 🫁 LumaCXR: Pneumonia Diagnostic AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**LumaCXR** is a state-of-the-art, web-based diagnostic terminal powered by a Custom Convolutional Neural Network (CNN). It is designed to analyze frontal pediatric chest X-rays and detect visual anomalies consistent with focal consolidation or fluid buildup (Pneumonia) with high accuracy.

---

## 🔗 Quick Links
* **🌐 Live Web App:** [Play with LumaCXR on Streamlit]([INSERT_STREAMLIT_APP_LINK_HERE])
* **📓 Kaggle Notebook:** [View Model Training & Pipeline](https://www.kaggle.com/code/youssefamgadelkhatib/lumacxr)
* **🗄️ Kaggle Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## 📊 Model Performance
The LumaCXR custom architecture was rigorously evaluated on a holdout test set of 587 patient X-rays, achieving an overall **Accuracy of 98.3%**. 

| Classification | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Normal (Clear)** | 95% | 98% | 97% | 149 |
| **Pneumonia** | 99% | 98% | 99% | 438 |

**Aggregate Metrics (Weighted):**
* **Precision:** 98.32%
* **Recall:** 98.30%
* **F1-Score:** 98.30%

---

## ✨ Key Features
* **Custom CNN Architecture:** Built from scratch to prevent overfitting and specifically target pulmonary opacities, avoiding the domain-gap issues of generic pre-trained models.
* **High-Tech Clinical UI:** A custom-themed dark mode interface with cyan accents, simulating a real hospital PACS (Picture Archiving and Communication System) terminal.
* **Interactive Diagnostics:** Features a simulated scanning animation and digital dashboard metrics that provide instant AI confidence scores.
* **Edge-Case Resilience:** Trained using extensive data augmentation (rotations, zooms, and shifts) and heavy dropout layers to ensure robust, generalized predictions.

---

## 🛠️ Tech Stack
* **Deep Learning Framework:** TensorFlow / Keras
* **Web Frontend Framework:** Streamlit
* **Data Processing:** NumPy, Pillow (PIL)
* **Visualization:** Matplotlib / Seaborn (in training notebooks)

---
