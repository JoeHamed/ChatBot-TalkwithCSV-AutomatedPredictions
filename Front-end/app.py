import streamlit as st

backend_url = "http://localhost:8000/"

# ----- Page Setup -----
observe_data_page = st.Page(
    page="views/observe_data.py",
    title="Observe Data",
    icon='📊',
    default=True
)
chatbot_page = st.Page(
    page="views/chatbot.py",
    title="AI-Assistant",
    icon='🤖'
)
manual_prediction_page = st.Page(
    page="views/manual_prediction.py",
    title="Manual Prediction",
    icon='🤔'
)


# ----- Navigation Menu -----
nav = st.navigation(
    {
        "Observation & Manual Prediction": [observe_data_page, manual_prediction_page],
        "Chatbot": [chatbot_page],
    }
)

# ----- Run Navigation -----
nav.run()

# Shared on all pages
st.logo('assets/logo.png', size='large')