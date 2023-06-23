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
from langchain.memory import VectorStoreRetrieverMemory
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS

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
        uploaded_file = st.file_uploader("Upload your files and click on 'Process'")
        # Extract text and chunk and vectorize it
        # Store vectors (vectorstore)


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

    # TODO Call the model and store & display response & handle user input
    ######################################################
    if uploaded_file and user_input:
        article = uploaded_file.read().decode()
        # Extract text and chunk and vectorize it
        embeddings = OpenAIEmbeddings()
        query_result = embeddings.embed_query(user_input)
        doc_result = embeddings.embed_documents([article])

        # Store vectors (vectorstore)
        embedding_size = 1536 #text-embedding-ada-002 model dimension size magic number
        index = faiss.IndexFlatL2(embedding_size)
        embedding_fn = OpenAIEmbeddings().embed_query(user_input)
        vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
        #TODO FINISH THIS PART by creating a vectorStoreRetrieverMemory
        retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
        memory = VectorStoreRetrieverMemory(retriever=retriever)

        prompt = HumanMessage(content=user_input)
        st.session_state.messages.append(prompt)
        with st.spinner("Thinking..."):
            response = chat(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))

        # display message history
        messages = st.session_state.get('messages', [])
        for i, msg in enumerate(messages[1:]):
            if i % 2 == 0:
                message(msg.content, is_user=True, key=str(i) + '_user')
            else:
                message(msg.content, is_user=False, key=str(i) + '_ai')

    ###################################################################

    # handle user input
    # if user_input:
    #     st.session_state.messages.append(HumanMessage(content=user_input))
    #     with st.spinner("Thinking..."):
    #         response = chat(st.session_state.messages)
    #     st.session_state.messages.append(AIMessage(content=response.content))


    # Manage context (memory)




if __name__ == '__main__':
    main()
