# ❤️ Heart Disease Prediction System

<p align="center">
  <strong>Machine Learning • KNN • Streamlit • Python</strong>
</p>

<p align="center">
  An interactive machine learning application for predicting the likelihood of heart disease from patient health parameters.
</p>

---

## 📌 About the Project

The **Heart Disease Prediction System** is a machine learning web application built with **Python and Streamlit**.

The application uses a trained **K-Nearest Neighbors (KNN)** classification model to analyze patient health and cardiovascular parameters and generate a heart disease risk prediction.

The trained model, feature scaler, and expected input columns are saved using **Joblib** and loaded directly into the Streamlit application.

> ⚠️ **Note:** This project is intended for educational and demonstration purposes and should not be used as a substitute for professional medical diagnosis.

---

## ✨ Features

- 🫀 Interactive patient health information form
- 🤖 K-Nearest Neighbors (KNN) prediction model
- 📊 Multiple cardiovascular and health parameters
- ⚙️ Feature scaling using `StandardScaler`
- 💾 Persistent ML model using Joblib
- 🎨 Modern Streamlit interface
- ⚡ Instant prediction results
- ✅ Low-risk prediction result
- ⚠️ High-risk prediction result
- 📱 Responsive user interface

---

## 🖥️ Application Preview

### 🏠 Heart Disease Prediction Dashboard

The application provides an interactive interface where users can enter patient health and cardiovascular information.

![Heart Disease Prediction Dashboard](screenshots/dashboard.png)

### ✅ Prediction Result

After submitting the patient information, the trained KNN model generates a heart disease risk prediction.

![Heart Disease Prediction Result](screenshots/prediction.png)

---

## 🧠 Machine Learning

### Algorithm

The project uses the **K-Nearest Neighbors (KNN)** classification algorithm.

### Preprocessing

The input data is processed using:

- Feature scaling with `StandardScaler`
- One-hot encoded categorical variables
- Consistent feature ordering using saved expected columns

### Model Files

| File | Purpose |
|---|---|
| `knn_heart_model.pkl` | Trained KNN classification model |
| `heart_scaler.pkl` | Feature scaling object |
| `heart_columns.pkl` | Expected model input columns |

---

## 📋 Input Parameters

The application collects the following information:

| Parameter | Description |
|---|---|
| Age | Patient's age |
| Sex | Patient's sex |
| Chest Pain Type | Type of chest pain |
| Resting BP | Resting blood pressure |
| Cholesterol | Serum cholesterol level |
| Fasting Blood Sugar | Whether fasting blood sugar is above 120 mg/dL |
| Resting ECG | Resting electrocardiogram result |
| Max Heart Rate | Maximum heart rate achieved |
| Exercise Angina | Exercise-induced angina |
| Oldpeak | ST depression value |
| ST Slope | Slope of the peak exercise ST segment |

---

## 🔄 How It Works

```text
                Patient Information
                        │
                        ▼
                Data Preprocessing
                        │
                        ▼
                 Feature Encoding
                        │
                        ▼
                  Feature Scaling
                        │
                        ▼
                KNN Classification
                        │
                        ▼
                 Risk Prediction
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
          Low Risk             High Risk
```

---

## 🛠️ Tech Stack

### Programming Language

- 🐍 Python

### Machine Learning

- Scikit-learn
- K-Nearest Neighbors
- StandardScaler

### Data Processing

- Pandas
- NumPy

### Model Persistence

- Joblib

### Web Application

- Streamlit

---

## 📂 Project Structure

```text
heart-disease-prediction/
│
├── screenshots/
│   ├── dashboard.png
│   └── prediction.png
│
├── app.py
├── heart.csv
│
├── knn_heart_model.pkl
├── heart_scaler.pkl
├── heart_columns.pkl
│
├── Heart_Prediction_Model.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- pip
- Git

### 1️⃣ Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd heart-disease-prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser.

If it doesn't open automatically, visit:

```text
http://localhost:8501
```

---

## 📈 Model Development

The machine learning workflow includes:

1. Loading the heart disease dataset
2. Exploring and understanding the data
3. Preparing numerical and categorical features
4. Encoding categorical variables
5. Scaling numerical features
6. Training the KNN classifier
7. Evaluating the model
8. Saving the trained model
9. Integrating the model with Streamlit

The complete model development process is available in:

```text
Heart_Prediction_Model.ipynb
```

---

## 🔮 Future Improvements

- 📊 Display prediction probability
- 📈 Add model performance visualizations
- 🧮 Compare KNN with Logistic Regression, Random Forest, and SVM
- 📋 Add detailed model evaluation metrics
- 📉 Add confusion matrix and ROC curve
- 📊 Add interactive health-data visualizations
- 🌐 Deploy the application online
- 📱 Further improve mobile responsiveness

---

## ⚠️ Disclaimer

This application is developed strictly for **educational and demonstration purposes**.

The prediction generated by this system should **not** be considered a medical diagnosis or used as a replacement for consultation with a qualified healthcare professional.

---

## 👨‍💻 Author

### Rishi

**Aspiring Data Analyst | Machine Learning Enthusiast | Python Developer**

Interested in building practical solutions using:

- Python
- Data Analytics
- Machine Learning
- SQL
- Streamlit

---

## ⭐ If You Like This Project

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub!

---
