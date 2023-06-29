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
import pypdf
import docx2txt
import variables

def init():
    """Sets API Key"""
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

def pdf_to_pages(file):
	"""extract text (pages) from pdf file"""
	pages = []
	pdf = pypdf.PdfReader(file)
	for p in range(len(pdf.pages)):
		page = pdf.pages[p]
		text = page.extract_text()
		pages += [text]
	return pages

def create_retriever(uploaded_file):
    """returns a retriever for the given uploaded_file 
    uploaded_file: txt, pdf, docx"""
    #TODO: doc, xls, zip
    # Extract text
    documents = []
    for f in uploaded_file:
        file_name = f.name
        if file_name.lower().endswith(".pdf"):
            documents.append(pdf_to_pages(f)[0])
        elif file_name.lower().endswith(".docx"):
            documents.append(docx2txt.process(f))
        elif file_name.lower().endswith(".txt"):
            documents.append(f.read().decode())
        else:
            print("unsupported file type! :(")
    # Break text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents(documents)
    # Store vectors (vectorstore)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    # Create retriever interface
    return db.as_retriever()

def generate_response(chat, uploaded_file, user_input, retriever):
    """returns a response using the retriever and user_input and chat history"""
    if uploaded_file and user_input:
        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=chat, chain_type='stuff', retriever=retriever)
        return qa.run(user_input)

# main fn
def main():
    # load API Key
    init()
    chat = ChatOpenAI(temperature=0)

    # the left sidebar section
    with st.sidebar:
        st.title("Your documents")
        
        #handle clicking 'process'
        if 'clicked' not in st.session_state:
            st.session_state.clicked = False
        def click_button():
            st.session_state.clicked = True
            variables.num_files = len(uploaded_file)
            print(variables.num_files)
        def unclick_button():
            st.session_state.clicked = False
            variables.num_files = len(uploaded_file)
            print(variables.num_files)
        # Upload pdf box and display upload document on screen
        uploaded_file = st.file_uploader("Upload your files and click on 'Process'", 
                                         accept_multiple_files = True, on_change=unclick_button)

        #create 'Process' buttond
        st.button("Process", on_click=click_button)

    # the right main section
    st.header("Chat with Multiple Documents 🤖")

    # Capture User's prompt
    with st.form("prompt form", clear_on_submit=True):
        print(variables.num_files)
        user_input = st.text_input("Ask a question about your documents: ", key="user_input", 
                    placeholder="Can you give me a short summary?", 
                    disabled=((not uploaded_file) or (variables.num_files == 0)))
        st.form_submit_button("Enter", use_container_width=True)

    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    # generate GPT's response 
    if user_input and st.session_state.clicked:
            prompt = HumanMessage(content=user_input)
            st.session_state.messages.append(prompt)
            # clears input after user enters prompt
            with st.spinner("Thinking..."):
                response = generate_response(chat, uploaded_file, user_input,
                                              create_retriever(uploaded_file))
            st.session_state.messages.append(AIMessage(content=response))
    
    # chat history
    with st.container():
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
