# import necessary libraries
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


def init():
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    

# main fn
def main():
    # load API Key
    init()
    chat = ChatOpenAI(temperature=0)

    # the left sidebar section
    with st.sidebar:
        st.title("Your documents")
        # Upload pdf box and dispaly upload document on screen
        uploaded_file = st.file_uploader("Upload your files and click on 'Process'", accept_multiple_files = True)
        #create 'Process' button
        process = st.button("Process")
        if process:
            # Extract text
            documents = []
            for f in uploaded_file:
                documents.append(f.read().decode())
            # Break text into chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.create_documents(documents)
            # embed
            embeddings = OpenAIEmbeddings()
            # Store vectors (vectorstore)
            db = Chroma.from_documents(texts, embeddings)

    # the right main section
    # Display header on screen
    st.header("Chat with Multiple Documents 🤖")
    # Capture User's prompt
    user_input = st.text_input("Ask a question about your documents: ", key="user_input", 
                placeholder="Can you give me a short summary?", disabled=not uploaded_file)
    
    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    def generate_response(uploaded_file, user_input, db):
        if uploaded_file and user_input:
            # Create retriever interface
            retriever = db.as_retriever()
            # Create QA chain
            qa = RetrievalQA.from_chain_type(llm=chat, chain_type='stuff', retriever=retriever)
            return qa.run(user_input)

    if user_input and process:
        prompt = HumanMessage(content=user_input)
        st.session_state.messages.append(prompt)
        with st.spinner("Thinking..."):
            #response = chat(st.session_state.messages)
            response = generate_response(uploaded_file, user_input, db)
        # st.session_state.messages.append(AIMessage(content=response.content))
        st.session_state.messages.append(AIMessage(content=response))

    # display message history
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')

    # Manage context (memory)




if __name__ == '__main__':
    main()
