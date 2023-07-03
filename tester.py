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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import pypdf
import docx2txt
import variables
import json

embeddings=OpenAIEmbeddings()

class Object:
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)
    
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


def embed(uploaded_file, embeddings):
    """returns a retriever for the given uploaded_file 
    uploaded_file: txt, pdf, docx"""
    #TODO: doc, xls, zip
    # Extract text
    documents = []
    metadatas=[]
    ids = []
    id = str(1)
    for f in uploaded_file:
        file_name = f.name
        metadatas.append({"file_name":file_name})
        ids.append(id)
        id= str(int(1)+1)
        if file_name.lower().endswith(".pdf"):
            documents.append(pdf_to_pages(f)[0])
        elif file_name.lower().endswith(".docx"):
            documents.append(docx2txt.process(f))
        elif file_name.lower().endswith(".txt"):
            documents.append(f.read().decode())
        else:
            print("unsupported file type! :(")
    # Break text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=[" ", ",", "\n"])
    texts = text_splitter.create_documents(documents)
    print("!!!!!!!!!!!!!!!!!TEXTS!!!!!!!!!!!!!!!!!!!!!")
    print(texts)

    
    db = Chroma.from_documents(texts, embeddings, collection_name="uploaded_files",
                                persist_directory="./chroma_db")
    db.persist()
    print("!!!!!!!!!!!!!!!!!DB.GET!!!!!!!!!!!!!!!!!!!!!")
    # print(db.get(include=['embeddings', 'documents', 'metadatas']))
    print(db.get())
    return db

def get_similar_text(user_input):
    # # load from local 
    db3 = Chroma(persist_directory="./chroma_db", collection_name="uploaded_files", embedding_function=embeddings)
    print("!!!!!!!!!!!!!!!!!DB3.GET RIGHT BEFORE ASKING FOR!!!!!!!!!!!!!!!!!!!!!")
    print(db3.get())
    res = db3.similarity_search(user_input, k = 2)
    
    print("!!!!!!!!!!!!!!!!!RESSS!!!!!!!!!!!!!!!!!!!!!")
    print(res)
    
    print(res[0].page_content)
    return res[0].page_content
        
db = Chroma(persist_directory="./chroma_db", collection_name="uploaded_files", embedding_function=OpenAIEmbeddings())

# main fn
def main():
    # load API Key
    init()
    chat = ChatOpenAI(temperature=0)
    embeddings=OpenAIEmbeddings()
    

    # the left sidebar section
    with st.sidebar:
        st.title("Your documents")
        
        #handle clicking 'process'
        if 'clicked' not in st.session_state:
            st.session_state.clicked = False

        # Upload pdf box and display upload document on screen
        uploaded_file = st.file_uploader("Upload your files and click on 'Process'", 
                                         accept_multiple_files = True)
        if uploaded_file !=None:
            #create 'Process' button
            if st.button("Process"):
                variables.db = embed(uploaded_file, embeddings)

    # the right main section
    st.header("Chat with Multiple Documents 🤖")
    # Capture User's prompt
    with st.form("prompt form", clear_on_submit=True):
        print(variables.num_files)
        user_input = st.text_input("Ask a question about your documents: ", key="user_input", 
                    placeholder="Can you give me a short summary?")
        enter_button = st.form_submit_button("Enter", use_container_width=True)

    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
    
    # Manage context (memory)
    conversation = ConversationChain(
        llm = chat,
        verbose = True,
        memory = ConversationBufferMemory()
    )

    # generate GPT's response 
    if user_input and enter_button:
            prompt = HumanMessage(content=user_input)
            st.session_state.messages.append(prompt)
            # clears input after user enters prompt
            with st.spinner("Thinking..."):
                extracted_text = get_similar_text(user_input)
                final_prompt = ("Answer the question based on the text below and nothing else. Text: ###"
                                +str(extracted_text)+" ### and the question: ###"+str(user_input)+" ###")
                response = conversation.predict(input=final_prompt) 
            st.session_state.messages.append(AIMessage(content=str(response)))
    
    # chat history
    with st.container():
        # display message history
        messages = st.session_state.get('messages', [])
        for i, msg in enumerate(messages[1:]):
            if i % 2 == 0:
                message(msg.content, is_user=True, key=str(i) + '_user')
            else:
                message(msg.content, is_user=False, key=str(i) + '_ai')

if __name__ == '__main__':
    main()