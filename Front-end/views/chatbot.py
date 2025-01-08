import streamlit as st
import requests

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


# Page Configuration
st.set_page_config(page_title='AI-Assistant', page_icon='🤖', layout='centered')

# Styling
st.markdown(h1_style, unsafe_allow_html=True)

st.title("AI-Assistant 🤖")

def check_file():
    check_file = requests.get(url=f'{backend_url}check_file/')
    if check_file.status_code == 200:
        # parse json response
        check_file = check_file.json()
        # access specific values
        message = check_file.get('Message')
        if message == 'No csv file was uploaded':
            st.warning("⚠️ Please Upload a File From the Observe Data Tab")
        else:
            st.success(f'✅ File ({message}) was uploaded')
            return message

def result_and_update_chat(prompt, history):
    chat = {"question": prompt, "chat_history": history}
    result = requests.post(url=f'{backend_url}/get_chain_result/', json=chat)
    # parse json response
    result = result.json()
    # access specific values
    completion = result.get('answer')
    #st.session_state.history.append((prompt, completion))
    st.session_state.history.append({"prompt": prompt, "history": completion})  # Append dictionaries
    return completion

# Checking if there is an already uploaded file
csv_file_name = check_file()

if csv_file_name is not None:
    mode = st.radio(
        "Select the mode",
        ["Single Prediction", "Talk to csv"],
        captions=[
            "Make a Single Prediction by passing a phrase that has the features",
            "Ask a General Question about the csv file uploaded",
        ],)
    if mode == "Talk to csv":
        # Initialize Chat History
        if "history" not in st.session_state:
            st.session_state.history = []

        if "completion" not in st.session_state:
            st.session_state.completion = [f"Ask any question about {csv_file_name}"]

        if "prompt" not in st.session_state:
            st.session_state.prompts = []



        # Display Chat History
        for message in st.session_state.history:

            with st.chat_message("user", avatar='😃'):
                st.markdown(message["prompt"])

            with st.chat_message("assistant", avatar='🤖'):
                st.markdown(message["history"])

        # for message in st.session_state.history:
        #     with st.chat_message(message["prompt"]):
        #         st.markdown(message["history"])

        # React to user Input
        if prompt := st.chat_input("Message AI-Assistant"):
            # Display user messages in chat message container
            with st.chat_message("user", avatar='😃'):
                st.write(prompt)

            # Process the prompt and add user message to chat history
            with st.spinner("Processing your request..."):
                print(st.session_state.history)
                print(type(st.session_state.history))
                print(type(prompt))
                completion = result_and_update_chat(prompt, st.session_state.history)

            # Display AI messages in chat message container
            with st.chat_message("assistant", avatar='🤖'):
                st.write(completion)

    elif mode == "Single Prediction":
        # if prompt := st.chat_input("Describe your situation for prediction:"):
        #     with st.chat_message("user"):
        #         st.write(prompt)
        #
        #     with st.spinner("Extracting features and Generating a Prediction..."):
        #         # Use LLM to parse features from the input
        #         user_input = {"user_input": prompt}
        #         prediction = requests.post(url=f'{backend_url}/extract_features/validate/make_a_prediction/', json=user_input)
        #
        #     if prediction.status_code == 200:
        #         prediction = prediction.json()
        #
        #         if prediction.get('Prediction') > 0.5:
        #             st.chat_message("assistant").write(f"Customer will {round((prediction.get('Prediction')*100), 2)}% churn ❌")
        #         else:
        #             st.chat_message("assistant").write(f"Customer will {round((100-(prediction.get('Prediction')*100)), 2)}% not churn ✅")
        #     else:
        #         st.error(f"Something Wrong Happened : {prediction.status_code}")
        # Initialize Single Prediction History
        if "single_prediction_history" not in st.session_state:
            st.session_state.single_prediction_history = []

        # Display previous predictions
        for entry in st.session_state.single_prediction_history:
            with st.chat_message("user", avatar='😃'):
                st.write(entry["prompt"])
            with st.chat_message("assistant", avatar='🤖'):
                st.write(entry["prediction"])

        # React to user Input
        if prompt := st.chat_input("Describe your situation for prediction:"):
            with st.chat_message("user", avatar='😃'):
                st.write(prompt)

            with st.spinner("Extracting features and Generating a Prediction..."):
                # Use LLM to parse features from the input
                user_input = {"user_input": prompt}
                prediction = requests.post(url=f'{backend_url}/extract_features/validate/make_a_prediction/',
                                           json=user_input)

            if prediction.status_code == 200:
                prediction = prediction.json()

                if prediction.get('Prediction') > 0.5:
                    prediction_text = f"Customer will {round((prediction.get('Prediction') * 100), 2)}% churn ❌"
                else:
                    prediction_text = f"Customer will {round((100 - (prediction.get('Prediction') * 100)), 2)}% not churn ✅"

                # Display the prediction
                with st.chat_message("assistant", avatar='🤖'):
                    st.write(prediction_text)

                # Save the prompt and prediction in session state
                st.session_state.single_prediction_history.append({
                    "prompt": prompt,
                    "prediction": prediction_text
                })
            else:
                st.error(f"Something Wrong Happened : {prediction.status_code}")