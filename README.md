# [Click to See Project Showcase Video](https://drive.google.com/file/d/1nSljPgMcfcRY3Q8Rkes7L1wG49lFuFjC/view?usp=sharing)
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
  - Users can interact with the **Chatbot** to ask questions or seek help. The chatbot uses an LLM (e.g., OpenAI GPT or Hugging Face models) to answer queries related to the machine learning models, data preprocessing, or general questions.
  - **Chain Logic**: The chatbot leverages a chain of logic to fetch and respond based on various queries and integrate with the rest of the app.

- **Visualization**: View a heatmap of the confusion matrix and other metrics to evaluate model performance.

***Still working on the documentation. Check back soon!***

