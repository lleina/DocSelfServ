#import necessary libraries
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


def init():
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    

#main fn
def main():

    #load API Key
    init()
    chat = ChatOpenAI(temperature=0)
    #the right main section
    #Display header on screen
    st.header("Chat with Multiple Documents 🤖")
    #Capture User's prompt
    user_input = st.text_input("Your message: ", key="user_input")
    #Call the model and store & display response
    # handle user input
    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Thinking..."):
            response = chat(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))

    #Manage context (memory)
    #the left sidebar section
    with st.sidebar:
        st.header("🤖")
        #Display upload document on screen
        #Upload pdf box
        #Extract text and chunk and vectorize it
        #Store vectors (vectorstore)



if __name__ == '__main__':
    main()
