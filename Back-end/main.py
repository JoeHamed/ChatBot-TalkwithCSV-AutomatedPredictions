import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File
import pandas as pd
from pydantic import BaseModel
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from MLModels import MLModels
from ChurnPreProcessing import ChurnPreProcessing
from typing import Dict, Any
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from functools import lru_cache
import ast
import os
import re

app = FastAPI()

global_df = None # Get access to the dataset uploaded (No modifications)
global_filename = None # Get access to the name of the dataset uploaded
global_temp_path = None # Get the template file path
global_df_preprocessed = global_df
X_train, X_test, y_train, y_test = None, None, None, None # Get a hold of split data
llm_pipeline = None # Access the model pipeline

class TempFilePathRequest(BaseModel):
    temp_file_path: str

class PreProcessingRequest(BaseModel):
    preprocess: bool

class MLTypeParams(BaseModel):
    model: str
    params: Dict[str, Any]  # Flexible to handle various parameter types

DB_FAISS_PATH = 'vectorstore/db_faiss'
# sentence-transformers/all-distilroberta-v1
# sentence-transformers/all-MiniLM-L6-v2

@app.get("/") # Retrieve Info from server root
def read_root():
    return {"Message": "Data Visualiser Web App"}

@app.post("/uploadfile/") # Send the csv file to server
async def get_csv_file(file: UploadFile = File(...)):
    global global_df, global_filename
    try:
        global_df = pd.read_csv(file.file)
        global_filename = file.filename
        print(global_filename)
        num_rows, num_columns= global_df.shape

        return {"number_of_rows": num_rows,
                "number_of_columns": num_columns,
                "header": global_df.head().to_dict(orient='records'),}
    finally:
        pass

@app.get("/check_file/")
async def check_file():
    global global_df
    global global_filename
    if global_df is not None:
        return {"Message": global_filename}
    else:
        return {"Message": 'No csv file was uploaded'}

@app.post("/tempfilepath/")
async def temp_filepath(tmp_path: TempFilePathRequest):
    global global_temp_path
    global_temp_path = tmp_path.temp_file_path
    print(global_temp_path)
    return {"temp_file_path": global_temp_path}

@app.get("/tempfilepath/getpath/")
async def temp_filepath():
    global global_temp_path
    if global_temp_path is not None:
        return {"temp_file_path": global_temp_path}
    else:
        return {"temp_file_path": None}

@lru_cache()
def load_llm():
    global llm_pipeline
    if llm_pipeline is None:
        model_name = "meta-llama/Llama-2-7b-chat-hf"  # Adjust to your model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,  # Ensure input type matches compute dtype
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # Match the compute dtype
        )
        # Create a HuggingFace pipeline
        hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        # Return the pipeline wrapped in HuggingFacePipeline
        return HuggingFacePipeline(pipeline=hf_pipeline)
    else:
        return llm_pipeline


def llm_chain():
    llm = load_llm()  # Use the proper LangChain LLM instance
    if global_df is not None:
        if global_temp_path is not None:
            loader = CSVLoader(file_path=global_temp_path, encoding="utf-8", csv_args={'delimiter': ','})
            data = loader.load()

            # Create vector embeddings
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            embeddings = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                model_kwargs={'device': device}
            )
            # Split the data into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Adjust this value to reduce chunk size
                chunk_overlap=50  # Optional: overlap between chunks for better context
            )
            documents = text_splitter.split_documents(data)

            # Create FAISS index
            db = FAISS.from_documents(documents, embeddings)  # FAISS vector store from data using the embeddings
            db.save_local(DB_FAISS_PATH) # save the embeddings and the index for future use (No re-computing)

            # Define retrieval chain
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,  # Pass the correct LangChain-compatible LLM instance
                retriever=db.as_retriever(),
                #max_tokens_limit=256,

            )
            return chain

def generate_response(model, tokenizer, query, max_length=256, temperature=0.2):
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length, # max new tokens
        temperature=temperature, # Creativity of the response
        pad_token_id=tokenizer.eos_token_id  # Handle models without a pad token
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def make_prediction(features):
    # Load the classification model
    file_path = './model.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
    else:
        print("File not found!")

    # Convert entry dictionary to a pandas DataFrame (single row)
    features_df = pd.DataFrame([features])

    # Make a prediction
    prediction = model.predict(features_df)

    print(type(prediction))
    print(prediction)

    return prediction

def validate_and_complete_features(features):
    required_features = {
        "gender": 0,
        "Senior_Citizen": 0,
        "Is_Married": 0,
        "Dependents": 0,
        "tenure": 0.0,
        "Phone_Service": 0,
        "Dual": 0,
        "Online_Security": 0,
        "Online_Backup": 0,
        "Device_Protection": 0,
        "Tech_Support": 0,
        "Streaming_TV": 0,
        "Streaming_Movies": 0,
        "Paperless_Billing": 0,
        "Monthly_Charges": 0.0,
        "Total_Charges": 0.0,
        "Internet_Service_DSL": False,
        "Internet_Service_Fiber optic": False,
        "Internet_Service_No": False,
        "Contract_Month-to-month": False,
        "Contract_One year": False,
        "Contract_Two year": False,
        "Payment_Method_Bank transfer (automatic)": False,
        "Payment_Method_Credit card (automatic)": False,
        "Payment_Method_Electronic check": False,
        "Payment_Method_Mailed check": False,
    }
    # Fill in missing keys with default values
    for key, default_value in required_features.items():
        features.setdefault(key, default_value)

    # # Ensure all boolean values are converted to integers (True becomes 1, False becomes 0)
    # features = {key: int(value) if isinstance(value, bool) else value for key, value in features.items()}

    return features


@app.post("/extract_features/validate/make_a_prediction/")
def extract_features_with_llm(user_input: dict):
    user_input = user_input.get('user_input')
    # Load the llm
    llm = load_llm()

    llm_prompt = f"""
    Extract the following features from the given text. Provide the extracted features in JSON format.

    Features to extract:
    - **gender**: 0 for male, 1 for female
    - **Senior_Citizen**: 0 for no, 1 for yes
    - **Is_Married**: 0 for no, 1 for yes
    - **Dependents**: 0 for no, 1 for yes
    - **tenure**: Number of months the user has been with the service (numeric)
    - **Phone_Service**: 0 for no, 1 for yes
    - **Dual**: 0 for no, 1 for yes
    - **Online_Security**: 0 for no, 1 for yes
    - **Online_Backup**: 0 for no, 1 for yes
    - **Device_Protection**: 0 for no, 1 for yes
    - **Tech_Support**: 0 for no, 1 for yes
    - **Streaming_TV**: 0 for no, 1 for yes
    - **Streaming_Movies**: 0 for no, 1 for yes
    - **Paperless_Billing**: 0 for no, 1 for yes
    - **Monthly_Charges**: Numeric value
    - **Total_Charges**: Numeric value
    - **Internet_Service_DSL**: True or False
    - **Internet_Service_Fiber optic**: True or False
    - **Internet_Service_No**: True or False
    - **Contract_Month-to-month**: True or False
    - **Contract_One_year**: True or False
    - **Contract_Two_year**: True or False
    - **Payment_Method_Bank_transfer (automatic)**: True or False
    - **Payment_Method_Credit_card (automatic)**: True or False
    - **Payment_Method_Electronic_check**: True or False
    - **Payment_Method_Mailed_check**: True or False

    Input Text: {user_input}
    """

    # Call LLM
    llm_response = llm.generate([llm_prompt])
    print(llm_response)
    print(type(llm_response))

    extracted_features = None

    if hasattr(llm_response, 'generations'):
        for i, generation in enumerate(llm_response.generations):
            print(f"Response {i + 1}:")
            for gen in generation:  # Each `generation` is typically a list of outputs
                # Use regex to find the first dictionary-like structure in the text
                match = re.search(r"\{.*?\}", gen.text, re.DOTALL)
                if match:
                    dict_str = match.group(0)
                    print("Matched Text:", dict_str)
                    # Replace potential JSON issues (e.g., true/false)
                    dict_str = dict_str.replace("false", "False").replace("true", "True")
                    try:
                        # Safely parse the dictionary string to an actual Python dictionary
                        extracted_features = ast.literal_eval(dict_str)
                        print("Extracted Features:", extracted_features)

                    except (SyntaxError, ValueError):
                        print("Error parsing dictionary:", dict_str)

    print(type(extracted_features))

    print(extracted_features)
    #print(type(features))
    print("Shape of extracted_features:", np.array(extracted_features).shape)
    print("Type of extracted_features:", type(extracted_features))

    # Convert the dictionary values into a 2D array (i.e., a list of lists)
    array_data = np.array(list(extracted_features.values())).reshape(1, -1)
    array_data = np.squeeze(array_data)
    print(type(array_data))  # Should be <class 'list'>
    print(f"Length of values_list: {len(array_data)}")  # Should be 26

    # Load the classification model and make a prediction
    prediction = make_prediction(array_data)

    # Convert NumPy array to a list and access the first and only element
    prediction = prediction.tolist()[0][0]
    print(f"Prediction: {prediction}")

    return {'Prediction': prediction}



@app.post("/get_chain_result/")
async def get_chain_result(chat: dict):
    # parse json response
    # chat = chat.json()
    print(chat)
    print(type(chat))

    # Ensure the 'chat_history' is in the expected format
    question = chat.get("question")
    chat_history = chat.get("chat_history", [])

    # Ensure that chat_history is in a list of tuples (user_message, assistant_response)
    if isinstance(chat_history, list):
        # If chat_history is not already a list of tuples, convert it
        chat_history = [(message['prompt'], message['history']) for message in chat_history]

    chain = llm_chain()
    if chain is not None:
        # Now call the chain with the structured question and chat_history
        result = chain({"question": question, "chat_history": chat_history})
        #result = chain(chat)
        print(result)
        return result

@app.post("/ChurnUseCase/preprocessing/")
async def preprocessing(preprocess: PreProcessingRequest):
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = ChurnPreProcessing.preprocess(data=global_df)
    # Convert dataframes to numpy arrays to lists to ensure they can be serialized
    X_train = np.array(X_train).tolist()
    X_test = np.array(X_test).tolist()
    y_train = np.array(y_train).tolist()
    y_test = np.array(y_test).tolist()
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, 'y_test': y_test}


@app.post("/classification_method/")
async def classification_method(mltype_params: MLTypeParams):
    model, params = mltype_params.model, mltype_params.params
    print(model)
    print(params)
    if model == 'xgboost':
        print('using xgboost ..')
        cm, acc_score, k_cross_score, report = MLModels.xgboost_model(X_train, X_test, y_train, y_test, params=params)
        #<class 'numpy.ndarray'> <class 'numpy.float64'> <class 'numpy.ndarray'> <class 'str'>
        return {'cm': cm.tolist(), 'acc_score': float(acc_score), 'k_cross_score':k_cross_score.tolist(),
                'report': report}

    elif model == 'catboost':
        print('using catboost ..')
        cm, acc_score, k_cross_score, report = MLModels.catboost_model(X_train, X_test, y_train, y_test, params=params)
        # <class 'numpy.ndarray'> <class 'numpy.float64'> <class 'numpy.ndarray'> <class 'str'>
        return {'cm': cm.tolist(), 'acc_score': float(acc_score), 'k_cross_score': k_cross_score.tolist(),
                'report': report}

    elif model == 'randomforest':
        print('using random forest ..')
        cm, acc_score, k_cross_score, report = MLModels.random_forest_model(X_train, X_test, y_train, y_test, params=params)
        # <class 'numpy.ndarray'> <class 'numpy.float64'> <class 'numpy.ndarray'> <class 'str'>
        return {'cm': cm.tolist(), 'acc_score': float(acc_score), 'k_cross_score': k_cross_score.tolist(),
                'report': report}

    elif model == 'logistic':
        print('using logistic regression ..')
        cm, acc_score, k_cross_score, report = MLModels.logistic_regression_model(X_train, X_test, y_train, y_test, params=params)
        # <class 'numpy.ndarray'> <class 'numpy.float64'> <class 'numpy.ndarray'> <class 'str'>
        return {'cm': cm.tolist(), 'acc_score': float(acc_score), 'k_cross_score': k_cross_score.tolist(),
                'report': report}

    elif model == 'kernelsvm':
        print('using kernelsvm ..')
        cm, acc_score, k_cross_score, report = MLModels.kernel_svm_model(X_train, X_test, y_train, y_test,
                                                                                  params=params)
        # <class 'numpy.ndarray'> <class 'numpy.float64'> <class 'numpy.ndarray'> <class 'str'>
        return {'cm': cm.tolist(), 'acc_score': float(acc_score), 'k_cross_score': k_cross_score.tolist(),
                'report': report}
    elif model == 'ann':
        print('using ANN ..')
        cm, acc_score, report = MLModels.tensorflow_model(X_train, X_test, y_train, y_test,
                                                                         params=params)
        # <class 'numpy.ndarray'> <class 'numpy.float64'> <class 'str'>
        return {'cm': cm.tolist(), 'acc_score': float(acc_score),'report': report}




if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
