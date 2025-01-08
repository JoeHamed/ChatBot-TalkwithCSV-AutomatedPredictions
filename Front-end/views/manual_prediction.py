import numpy as np
import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit import spinner

backend_url = "http://localhost:8000/"

h1_style = """
<style>
   @keyframes slide-in {
      0% {
         opacity: 0;
         transform: translateY(-20px);
      }
      100% {
         opacity: 1;
         transform: translateY(0);
      }
   }

   h1 {
      font-size: 16px;
      text-align: center;
      text-transform: capitalize;
      font-family: 'Arial', sans-serif;
      animation: slide-in 1s ease-out;
   }
</style>
"""
def check_file():
    check_file = requests.get(url=f'{backend_url}check_file/')
    if check_file.status_code == 200:
        # parse json response
        check_file = check_file.json()
        # access specific values
        message = check_file.get('Message')
        if message == 'No csv file was uploaded':
            st.warning("⚠️ Please Upload a File From the Observe Data Tab")
            return False
        else:
            st.success(f'✅ File ({message}) was uploaded')
            return True

def svm_params():
    st.sidebar.subheader('SVM Hyperparameters')

    C = st.sidebar.number_input('Regularization Parameter (C)', 0.01, 10.0, step=0.01, key='C_svm')
    kernel = st.sidebar.radio('Kernel', ('linear', 'poly', 'rbf', 'sigmoid'), key='kernel')
    st.sidebar.link_button('Show kernel Documentation', 'https://scikit-learn.org/1.5/modules/svm.html#svm-kernels')
    degree = st.sidebar.number_input('Degree (for poly kernel)', 1, 10, step=1, key='degree')
    gamma = st.sidebar.radio('Gamma (Kernel Coefficient)', ('scale', 'auto'), key='gamma_svm')
    tol = st.sidebar.number_input('Tolerance for Stopping Criteria (tol)', 1e-6, 1e-1, step=1e-3, format="%.1e",
                                  key='tol_svm')
    max_iter = st.sidebar.number_input('Maximum Number of Iterations (max_iter)', -1, 10000, step=100, key='max_iter_svm')

    params = {'C': C, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'tol': tol, 'max_iter': max_iter}
    return params


def logic_params():
    st.sidebar.subheader('Logistic Regression Hyperparameters')

    C = st.sidebar.number_input('Inverse Regularization Strength (C)', 0.01, 10.0, step=0.01, key='C_logic')
    max_iter = st.sidebar.number_input('Maximum Number of Iterations (max_iter)', 50, 1000, step=50, key='max_iter_logic')
    solver = st.sidebar.radio('Solver (solver)', ('lbfgs', 'liblinear', 'saga', 'newton-cg'), key='solver')
    penalty = st.sidebar.radio('Penalty (penalty)', ('l2', 'l1', 'elasticnet', 'none'), key='penalty')
    fit_intercept = st.sidebar.radio('Fit Intercept (fit_intercept)', (True, False), key='fit_intercept')
    tol = st.sidebar.number_input('Tolerance for Stopping Criteria (tol)', 1e-6, 1e-1, step=1e-3, format="%.1e",
                                  key='tol_logic')

    params = {'C': C, 'max_iter': max_iter, 'solver': solver, 'penalty': penalty, 'fit_intercept': fit_intercept,'tol': tol}
    return params

def forest_params():
    st.sidebar.subheader('Random Forest Hyperparameters')

    n_estimators = st.sidebar.number_input('Number of Estimators (n_estimators)', 10, 1000, step=10, key='n_estimators_forest')
    max_depth = st.sidebar.number_input('Max Depth (max_depth)', 1, 50, step=1, key='max_depth')
    min_samples_split = st.sidebar.number_input('Minimum Samples to Split (min_samples_split)', 2, 20, step=1,
                                                key='min_samples_split')
    min_samples_leaf = st.sidebar.number_input('Minimum Samples per Leaf (min_samples_leaf)', 1, 20, step=1,
                                               key='min_samples_leaf')
    max_features = st.sidebar.radio('Maximum Features (max_features)', ('sqrt', 'log2'), key='max_features')
    # instead of training on all the observations, each tree of RF is trained on a subset of the observations.
    bootstrap = st.sidebar.radio('Bootstrap Samples (bootstrap)', (True, False), key='bootstrap')

    params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf, 'max_features': max_features, 'bootstrap': bootstrap}
    return params


def tensorflow_ann_params():
    st.sidebar.subheader('TensorFlow ANN Hyperparameters')

    # Architecture Parameters
    num_layers = st.sidebar.number_input('Number of Hidden Layers', 1, 10, step=1, key='num_layers_ann')
    layer_configs = []  # To store configurations for each layer

    for i in range(num_layers):
        size = st.sidebar.number_input(
            f'Layer {i + 1} Size',
            1,
            1024,
            step=1,
            key=f'layer_{i + 1}_size_ann'
        )
        activation = st.sidebar.radio(
            f'Layer {i + 1} Activation',
            ('relu', 'sigmoid', 'tanh', 'softmax', 'linear'),
            key=f'layer_{i + 1}_activation_ann'
        )
        dropout_rate = st.sidebar.slider(
            f'Layer {i + 1} Dropout Rate',
            0.0,
            0.5,
            step=0.05,
            key=f'layer_{i + 1}_dropout_ann'
        )
        l2_regularization = st.sidebar.number_input(
            f'Layer {i + 1} L2 Regularization',
            0.0,
            0.1,
            step=0.001,
            format="%.3f",
            key=f'layer_{i + 1}_l2_reg_ann'
        )
        layer_configs.append({
            'size': size,
            'activation': activation,
            'dropout_rate': dropout_rate,
            'l2_regularization': l2_regularization
        })

    # Optimization Parameters
    optimizer = st.sidebar.radio(
        'Optimizer',
        ('adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta'),
        key='optimizer_ann'
    )
    learning_rate = st.sidebar.number_input(
        'Learning Rate',
        0.0001,
        1.0,
        step=0.0001,
        format="%.4f",
        key='learning_rate_ann'
    )

    # Training Parameters
    batch_size = st.sidebar.number_input(
        'Batch Size',
        16,
        512,
        step=16,
        key='batch_size_ann'
    )
    epochs = st.sidebar.number_input(
        'Number of Epochs',
        1,
        1000,
        step=1,
        key='epochs_ann'
    )

    # Display the selected layer configuration
    st.text("Layer Configuration:")
    for idx, config in enumerate(layer_configs):
        st.code(
            f"Layer {idx + 1}: Size = {config['size']}, "
            f"Activation = {config['activation']}, "
            f"Dropout = {config['dropout_rate']}, "
            f"L2 = {config['l2_regularization']}"
        )
    params = {'layer_configs': layer_configs, 'optimizer': optimizer, 'learning_rate': learning_rate,
              'batch_size': batch_size, 'epochs': epochs}
    return params


def xgb_params():
    st.sidebar.subheader('XGBoost Hyperparameters')

    learning_rate = st.sidebar.number_input('Learning Rate (eta)', 0.01, 1.0, step=0.01, key='learning_rate_xgb')
    n_estimators = st.sidebar.number_input('Number of Estimators (n_estimators)', 10, 1000, step=10, key='n_estimators_xgb')
    max_depth = st.sidebar.number_input('Max Depth', 1, 20, step=1, key='max_depth')
    subsample = st.sidebar.slider('Subsample', 0.5, 1.0, step=0.05, key='subsample')
    colsample_bytree = st.sidebar.slider('Colsample by Tree', 0.5, 1.0, step=0.05, key='colsample_bytree')
    gamma = st.sidebar.number_input('Gamma (min_split_loss)', 0.0, 10.0, step=0.1, key='gamma_xgb')
    reg_alpha = st.sidebar.number_input('L1 Regularization (reg_alpha)', 0.0, 1.0, step=0.01, key='reg_alpha')
    reg_lambda = st.sidebar.number_input('L2 Regularization (reg_lambda)', 0.0, 1.0, step=0.01, key='reg_lambda')

    params = {'learning_rate': learning_rate, 'n_estimators': n_estimators, 'max_depth': max_depth,
              'subsample': subsample, 'colsample_bytree': colsample_bytree, 'gamma': gamma, 'reg_alpha': reg_alpha,
              'reg_lambda': reg_lambda}
    return params


def cat_params():
    st.sidebar.subheader('CatBoost Hyperparameters')

    learning_rate = st.sidebar.number_input('Learning Rate (learning_rate)', 0.01, 1.0, step=0.01, key='learning_rate_cat')
    iterations = st.sidebar.number_input('Number of Iterations (iterations)', 100, 10000, step=100, key='iterations')
    depth = st.sidebar.number_input('Depth of Trees (depth)', 1, 16, step=1, key='depth')
    l2_leaf_reg = st.sidebar.number_input('L2 Leaf Regularization (l2_leaf_reg)', 0.0, 10.0, step=0.1,
                                          key='l2_leaf_reg')
    bagging_temperature = st.sidebar.slider('Bagging Temperature (bagging_temperature)', 0.0, 1.0, step=0.01,
                                            key='bagging_temperature')
    border_count = st.sidebar.number_input('Number of Splits for Quantization (border_count)', 1, 255, step=1,
                                           key='border_count')
    random_strength = st.sidebar.number_input('Random Strength (random_strength)', 0.0, 10.0, step=0.1,
                                              key='random_strength')
    params = {'learning_rate': learning_rate, 'iterations': iterations, 'depth': depth,
              'l2_leaf_reg': l2_leaf_reg, 'bagging_temperature': bagging_temperature,
              'border_count': border_count, 'random_strength': random_strength}
    return params

def json_to_dataframe(preprocess):
    # parse json response
    train_test_splits = preprocess.json()
    # access specific values
    X_train, X_test, y_train, y_test = (train_test_splits.get('X_train'),
                                        train_test_splits.get('X_test'),
                                        train_test_splits.get('y_train'),
                                        train_test_splits.get('y_test'))
    # Convert from list to np array to a pandas dataframe
    X_train, X_test, y_train, y_test = (pd.DataFrame(np.array(X_train)),
                                        pd.DataFrame(np.array(X_test)),
                                        pd.DataFrame(np.array(y_train)),
                                        pd.DataFrame(np.array(y_test)))
    return X_train, X_test, y_train, y_test

def metrics_taking_apart(metrics):
    cm = np.array(metrics.get('cm'))
    acc_score = metrics.get('acc_score')
    k_cross_score = np.array(metrics.get('k_cross_score'))
    class_report = metrics.get('report')
    return cm, acc_score, k_cross_score, class_report

def plotting_cm(cm):
    cm = np.array(cm)
    names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    counts = [value for value in cm.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names, counts, percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    fig, ax = plt.subplots()

    # Set background color
    # fig.patch.set_facecolor('#282A36')  # Background color for the entire figure
    # ax.set_facecolor('#282A36')  # Background color for the axis

    # Make the figure and axis background transparent
    fig.patch.set_alpha(0)  # Transparent figure background
    ax.set_facecolor('none')  # Transparent axis background

    sns.heatmap(cm, annot=labels, cmap='Purples', fmt='')
    st.pyplot(fig)

# Page Configuration
st.set_page_config(page_title='Manual Prediction', page_icon='🤔', layout='centered')

# Styling
st.markdown(h1_style, unsafe_allow_html=True)

st.title("Manual Prediction 🤔")

# Check if there is a file in the backend
file_check = check_file()



if file_check:
    # Sidebar layout
    st.subheader('Choose the Classifier')
    if st.toggle(label='⚙️ Apply preprocessing (Telco Customer Churn) to unlock', value=False):
        classifier = st.selectbox('Classifier', ('Support Vector Machine SVM', 'Logistic Regression', 'Random Forest',
                                                         'Artificial Neural Network ANN', 'XGBoost', 'CatBoost'))
        with st.spinner("Processing your request..."):
            preprocess = requests.post(url=f'{backend_url}/ChurnUseCase/preprocessing/', json={"preprocess": True})
            X_train, X_test, y_train, y_test = json_to_dataframe(preprocess)

        if preprocess.status_code == 200:
            if classifier == 'Support Vector Machine SVM':
                params = svm_params()
                if st.sidebar.button('Classify', key='classify'):
                    mltype_params = {'model': 'kernelsvm', 'params': params}
                    metrics = requests.post(url=f'{backend_url}classification_method/', json=mltype_params)
                    if metrics.status_code == 200:
                        cm, acc_score, k_cross_score, class_report = metrics_taking_apart(metrics.json())
                        st.metric(label='Accuracy Score', value=acc_score)
                        # Generate the plot using the Plotting class
                        plotting_cm(cm)
                        st.code(class_report)
                        st.code(f"K-Cross Validation Accuracy: {round(k_cross_score.mean() * 100, 2)} %")
                        st.code(f"Standard Deviation: {round(k_cross_score.std() * 100, 2)} %")

            elif classifier == 'Logistic Regression':
                params = logic_params()
                if st.sidebar.button('Classify', key='classify'):
                    mltype_params = {'model': 'logistic', 'params': params}
                    metrics = requests.post(url=f'{backend_url}classification_method/', json=mltype_params)
                    if metrics.status_code == 200:
                        cm, acc_score, k_cross_score, class_report = metrics_taking_apart(metrics.json())
                        st.metric(label='Accuracy Score', value=acc_score)
                        # Generate the plot using the Plotting class
                        plotting_cm(cm)
                        st.code(class_report)
                        st.code(f"K-Cross Validation Accuracy: {round(k_cross_score.mean() * 100, 2)} %")
                        st.code(f"Standard Deviation: {round(k_cross_score.std() * 100, 2)} %")

            elif classifier == 'Random Forest':
                params = forest_params()
                if st.sidebar.button('Classify', key='classify'):
                    mltype_params = {'model': 'randomforest', 'params': params}
                    metrics = requests.post(url=f'{backend_url}classification_method/', json=mltype_params)
                    if metrics.status_code == 200:
                        cm, acc_score, k_cross_score, class_report = metrics_taking_apart(metrics.json())
                        st.metric(label='Accuracy Score', value=acc_score)
                        # Generate the plot using the Plotting class
                        plotting_cm(cm)
                        st.code(class_report)
                        st.code(f"K-Cross Validation Accuracy: {round(k_cross_score.mean() * 100, 2)} %")
                        st.code(f"Standard Deviation: {round(k_cross_score.std() * 100, 2)} %")


            elif classifier == 'XGBoost':
                params = xgb_params()
                if st.sidebar.button('Classify', key='classify'):
                    mltype_params ={'model':'xgboost','params':params}
                    metrics = requests.post(url=f'{backend_url}classification_method/', json=mltype_params)
                    if metrics.status_code == 200:
                        cm, acc_score, k_cross_score, class_report = metrics_taking_apart(metrics.json())
                        st.metric(label='Accuracy Score', value=acc_score)
                        # Generate the plot using the Plotting class
                        plotting_cm(cm)
                        st.code(class_report)
                        st.code(f"K-Cross Validation Accuracy: {round(k_cross_score.mean()*100, 2)} %")
                        st.code(f"Standard Deviation: {round(k_cross_score.std()*100, 2)} %")


            elif classifier == 'CatBoost':
                params = cat_params()
                if st.sidebar.button('Classify', key='classify'):
                    mltype_params = {'model': 'catboost', 'params': params}
                    metrics = requests.post(url=f'{backend_url}classification_method/', json=mltype_params)
                    if metrics.status_code == 200:
                        cm, acc_score, k_cross_score, class_report = metrics_taking_apart(metrics.json())
                        st.metric(label='Accuracy Score', value=acc_score)
                        # Generate the plot using the Plotting class
                        plotting_cm(cm)
                        st.code(class_report)
                        st.code(f"K-Cross Validation Accuracy: {round(k_cross_score.mean() * 100, 2)} %")
                        st.code(f"Standard Deviation: {round(k_cross_score.std() * 100, 2)} %")


            elif classifier == 'Artificial Neural Network ANN':
                params = tensorflow_ann_params()
                if st.sidebar.button('Classify', key='classify'):
                    mltype_params = {'model': 'ann', 'params': params}
                    with st.spinner("Generating a Prediction..."):
                        metrics = requests.post(url=f'{backend_url}classification_method/', json=mltype_params)
                    if metrics.status_code == 200:
                        metrics = metrics.json()
                        cm, acc_score, class_report = np.array(metrics.get('cm')), metrics.get('acc_score'), metrics.get('report')
                        st.metric(label='Accuracy Score', value=acc_score)
                        # Generate the plot using the Plotting class
                        plotting_cm(cm)
                        st.code(class_report)

        else:
            pass
