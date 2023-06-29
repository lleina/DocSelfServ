# import necessary libraries
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os, uuid
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from chromadb.utils import embedding_functions
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
import chromadb
import openai
import pypdf
import docx2txt
import variables
import json
from bson import json_util
from openai.embeddings_utils import get_embedding,cosine_similarity
import numpy as np

    
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

def save_uploaded_file(uploaded_file):
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

def pdf_to_pages(file):
	"""extract text (pages) from pdf file"""
	pages = []
	pdf = pypdf.PdfReader(file)
	for p in range(len(pdf.pages)):
		page = pdf.pages[p]
		text = page.extract_text()
		pages += [text]
	return pages

def extract(file_name, uploaded_file):
    """returns a retriever for the given uploaded_file 
    uploaded_file: txt, pdf, docx"""
    #TODO: doc, xls, zip
    # Extract text
    
    #need to extract page by page rather than whole document later
    document = []
    if file_name.lower().endswith(".pdf"):
        content = pdf_to_pages(uploaded_file)[0]
    elif file_name.lower().endswith(".docx"):
        content = docx2txt.process(uploaded_file)
    elif file_name.lower().endswith(".txt"):
        content = uploaded_file.read().decode()

    obj = {
    "id": str(uuid.uuid4()),
    "text": content,
    "embedding": get_embedding(content,engine='text-embedding-ada-002')
    }
    document.append(obj)

    json_file_path = 'my_extracted_files.json'
    with open(json_file_path, 'r',encoding='utf-8') as f:
        data = json.load(f)

    for i in document:
            data.append(i)
    with open(json_file_path, 'w',encoding='utf-8') as f:
        json.dump(data, f,ensure_ascii=False, indent=4)

def model_response(user_input):
    user_query_vector = get_embedding(user_input,engine='text-embedding-ada-002')
    with open('my_extracted_files.json', 'r',encoding="utf-8") as jsonfile:
        data = json.load(jsonfile)
        for item in data:
            item['embeddings'] = np.array(item['embedding'])

        for item in data:
            item['similarities'] = cosine_similarity(item['embedding'], user_query_vector)
        sorted_data = sorted(data, key=lambda x: x['similarities'], reverse=True)

        context = ''
        for item in sorted_data[:2]:
            context += item['text']

        myMessages = [
            {"role": "system", "content": "You're a helpful Assistant."},
            {"role": "user", "content": "The following is a Context:\n{}\n\n Answer the following user query according to the above given context.\n\nquery: {}".format(context,user_input)}
        ]
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=myMessages,
            max_tokens=200,
        )

    return response['choices'][0]['message']['content']  
        
# main fn
def main():
    # load API Key
    init()
    chat = ChatOpenAI(temperature=0)

    # the left sidebar section
    with st.sidebar:
        st.title("Your documents")

        # Upload pdf box and display upload document on screen
        uploaded_file = st.file_uploader("Upload your files and click on 'Process'", 
                                         accept_multiple_files = True)
        if uploaded_file !=None:
            #create 'Process' button
            if st.button("Process"):
                for f in uploaded_file:
                    save_uploaded_file(f)
                    file_name = f.name
                    extract(file_name, f)

    # the right main section
    st.header("Chat with Multiple Documents 🤖")

    # Capture User's prompt
    with st.form("prompt form", clear_on_submit=True):
        print(variables.num_files)
        user_input = st.text_input("Ask a question about your documents: ", key="user_input", 
                    placeholder="Can you give me a short summary?", 
                   )
        st.form_submit_button("Enter", use_container_width=True)

    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
    
    # Manage context (memory)

    # generate GPT's response 
    if user_input:
        prompt = HumanMessage(content=user_input)
        st.session_state.messages.append(prompt)
        # clears input after user enters prompt
        with st.spinner("Thinking..."):
            response = model_response(user_input)
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
