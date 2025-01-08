import streamlit as st
import pandas as pd
import requests
import tempfile

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
            st.error('❌ No csv file was uploaded')
        else:
            st.success(f'✅ File ({message}) was uploaded')

def upload_file():
    uploaded_file = st.file_uploader("Upload File (CSV)", type=["csv"])

    if uploaded_file is not None:
        # use tempfile because CSVLoader only accepts a file_path
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name #tempfile path
        data = {"temp_file_path": tmp_file_path}
        # Send the file to the backend
        file_name = uploaded_file.name
        files = {'file': (file_name, uploaded_file.getvalue())} # file as bytes
        response = requests.post(url=f'{backend_url}uploadfile/', files=files)
        requests.post(url=f'{backend_url}tempfilepath/', json=data)

        if response.status_code != 200:
            st.error(response.text)
        else:
            st.success('File Uploaded', icon='✅')
            #st.write(response.json())
            data = response.json()
            header = data['header']
            df = pd.DataFrame(header)
            st.dataframe(df)
            st.write(f"📌 Dataset has {data['number_of_rows']} rows and {data['number_of_columns']} columns.")

def visualize_distribution():
    pass




# Page Configuration
st.set_page_config(page_title='Observe Data', page_icon='📊', layout='centered')

# Styling
st.markdown(h1_style, unsafe_allow_html=True)

# Title
st.title("Observe Data 📊")

# File Uploader
st.subheader('Upload a CSV file')
upload_file()

# Check if there is a file in the backend
check_file()
