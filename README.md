# [Click Here to See Project Showcase Video](https://drive.google.com/file/d/1nSljPgMcfcRY3Q8Rkes7L1wG49lFuFjC/view?usp=sharing)
# Web App with Chatbot Integration and Machine Learning Model Prediction 

This web application provides an interactive interface for uploading datasets, training machine learning models, and leveraging natural language processing through a chatbot powered by a large language model (LLM). Users can classify datasets, fine-tune model hyperparameters, and engage in conversation with the chatbot for assistance.

## Features

- **Model Selection**: Choose from multiple classifiers:
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Random Forest
  - Artificial Neural Network (ANN)
  - XGBoost
  - CatBoost

- **Hyperparameter Tuning**: Adjust model hyperparameters through the sidebar interface:
  - Regularization parameters, kernel selection, learning rate, number of estimators, and more.

- **Preprocessing**: Automatically preprocess the **Telco Customer Churn** dataset (or any dataset uploaded) and split it into training and testing sets.

- **Classification**: After choosing the model and hyperparameters, the app performs classification and displays:
  - **Accuracy Score**
  - **Confusion Matrix**
  - **K-Fold Cross Validation Accuracy**
  - **Classification Report**

- **Chatbot Integration**: 
  - Users can interact with the **Chatbot** to ask questions or seek help. The chatbot uses an LLM (Hugging Face models) to answer queries related to the machine learning models, data preprocessing, or general questions.
  - **Chain Logic**: The chatbot leverages a chain of logic to fetch and respond based on various queries and integrate with the rest of the app.

- **Visualization**: View a heatmap of the confusion matrix and other metrics to evaluate model performance.

## Structure
```bash
.
├── model.keras
├── model.pkl
├── requirements.txt
├── Showcase.mp4
├── Back-end
│   ├── ChurnPreProcessing.py
│   ├── main.py
│   ├── MLModels.py
├── data
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── Diagrams
│   ├── Manual_Predictions_Diagram.png
│   ├── Overall_Dataflow_Diagram.png
│   ├── SinglePrediction_Diagram.png
│   ├── Talktocsv_Diagram.png
├── Exploratory Data Analysis
│   ├── CatBoost Model Evaluation.pdf
│   ├── ChurnUseCase EDA.pdf
│   ├── ChurnUseCaseEDA.ipynb
│   ├── Initial thoughts and Models Experimenting.pdf
├── Final Model
│   ├── ChurnUseCase_TensorflowANN.ipynb
├── Front-end
│   ├── app.py
│   ├── .streamlit
│   │   ├── config.toml
│   ├── assets
│   │   ├── logo.png
│   ├── vectorstore
│   │   ├── db_faiss
│   │   │   ├── index.faiss
│   │   │   ├── index.pkl
│   ├── views
│       ├── chatbot.py
│       ├── manual_prediction.py
│       ├── observe_data.py
├── vectorstore
│   ├── db_faiss
│       ├── index.faiss
│       ├── index.pkl
```


# Data Flow Diagrams

<p align="center">
  <b>Overall Web App Data Pipeline Flow</b>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/d7c7aebe-3008-4c97-beda-a38eed8be29a" alt="Overall Web App Diagram" width="900" height="800">
</p>

<p align="center">
  <b>ChatBot 1st Mode [Single Prediction]</b> - <b>ChatBot 2nd Mode [Talk to CSV]</b>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/b1186337-bca3-41f9-9387-25a57fdfba36" alt="Single Prediction Diagram" width="335" height="600">
  <img src="https://github.com/user-attachments/assets/d3bb7859-f67b-4309-9e8c-dd0c80d4b04c" alt="Talktocsv Diagram" width="310" height="600">
</p>

<p align="center">
  <b>Manual Prediction Data Pipeline Flow</b>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/b3a7a1e8-d8cd-4910-b0e5-4f99ab528647" alt="Manual Prediction Diagram" width="400" height="660">
</p>


***Still working on the documentation. Check back soon!***


