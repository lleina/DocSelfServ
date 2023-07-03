import os
import streamlit as st
from json_functions import *
from dotenv import load_dotenv
#openai.api_key = '***'


def main():
    # Load the OpenAI API key from the environment variable
    #load_dotenv()
    
    st.title("PDF Repo - Self Service")

    uploaded_file = st.file_uploader("Choose a PDF file to upload", type="pdf")
    if uploaded_file is not None:
        if st.button("Read PDF"):
            save_uploaded_file(uploaded_file)
            with st.spinner("Processing..."):
                # st.write("Please wait while we learn the PDF.")
                learn_pdf(uploaded_file.name)
                # os.remove(uploaded_file.name)
                st.write("PDF reading completed! Now you may ask a question")               
    user_input = st.text_input("Enter your Query:")

    if st.button("Send"):
        with st.spinner("Creating response..."):
            st.write("You:", user_input)
            response = Answer_from_documents(user_input)
            st.write("Bot: "+response)


if __name__ == "__main__":
    main()