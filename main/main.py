import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os, uuid, PyPDF2, json, webbrowser, openai, docx2txt, copy
import aspose.words as aw
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from openai.embeddings_utils import get_embedding,cosine_similarity
import numpy as np
openai.api_key = ''

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

def extract(file_name):
    """returns a retriever for the given uploaded_file 
    uploaded_file: txt, pdf, docx"""
    #TODO: doc, xls, zip
    # Extract text
    
    #need to extract page by page rather than whole document later
    # document = []
    # if file_name.lower().endswith(".pdf"):
    #     content = pdf_to_pages(uploaded_file)[0]
    # elif file_name.lower().endswith(".docx"):
    #     content = docx2txt.process(uploaded_file)
    # elif file_name.lower().endswith(".txt"):
    #     content = uploaded_file.read().decode()
    #     def learn_pdf(file_path):
    
    content_chunks = []

    #for pdf
    pdf_file = open(file_name, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    page_number = 1
    for page in pdf_reader.pages:
        content = page.extract_text()
        # saves html file locally
        # aw.Document(file_name).save(file_name+".html")
        obj = {
            "id": str(uuid.uuid4()),
            "text": content,
            "page": page_number,
            "file_name": file_name,
            "embedding": get_embedding(content,engine='text-embedding-ada-002')
        }
        page_number +=1
        content_chunks.append(obj)

    # Save the learned data into the knowledge base. The json file must alread exist with just '[' and ']' and a blank line in between.
    # In this implementation embeddings for newly uploaded documents are appended to the json file...

    json_file_path = file_name+'.json'
    with open(json_file_path, 'a+',encoding='utf-8') as f:
        f.write("[ \n\n]")
    with open(json_file_path, 'r',encoding='utf-8') as f:
        data = json.load(f)
        
    for i in content_chunks:
            data.append(i)
    with open(json_file_path, 'w',encoding='utf-8') as f:
        json.dump(data, f,ensure_ascii=False, indent=4)
    
    pdf_file.close()


def model_response(user_input, file_names):
    user_query_vector = get_embedding(user_input,engine='text-embedding-ada-002')
    unsorted_data = []
    for file in file_names:
        with open(file+'.json', 'r',encoding="utf-8") as jsonfile:
            data = json.load(jsonfile)
            for item in data:
                item['embeddings'] = np.array(item['embedding'])
            for item in data:
                item['similarities'] = cosine_similarity(item['embedding'], user_query_vector)
                unsorted_data.append(item)
    sorted_data = sorted(unsorted_data, key=lambda x: x['similarities'], reverse=True)
    context = ''
    source = []
    for item in sorted_data[:3]:
        context += item['text']
        if (item['file_name'] + " page: " + str(item['page'])) not in source:
            source.append(item['file_name']+", page: " + str(item['page']))
    if context == '':
            context = 'There is NO CONTENT!'
    myMessages = [
        {"role": "system", "content": "You're a helpful Assistant."},
        {"role": "user", "content": "Answer the following QUERY:\n ### {} ###\n\n using the CONTENT:\n### {}### \n\n If the answer isn't found in the CONTENT provided, always respond with exactly this sentence: Sorry, the content does not contain that information. ".format(user_input,context)}
    ]
    print({"role": "user", "content": "Answer the following QUERY:\n ### {} ###\n\n. If the answer isn't found in the CONTENT provided, always respond with exactly this sentence: Sorry, the content does not contain that information. \n\n Here is the CONTENT:### {}###".format(user_input,context)})
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=myMessages,
        max_tokens=200,
    )
    #print(source)
    return [response['choices'][0]['message']['content'], source]

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
                    with open('file_names1.json', 'r', encoding='utf-8') as var:
                        file_names = json.load(var)
                    if file_name not in file_names[0]['file_names']:
                        file_names[0]['file_names'].append(file_name)
                        print(file_names)
                        with open('file_names1.json', 'w', encoding='utf-8') as var:
                            json.dump(file_names, var ,ensure_ascii=False, indent=4)
                        extract(file_name)

        #create a button to represent file
        with open('file_names1.json', 'r', encoding='utf-8') as var:
                        file_names = json.load(var)
        selected_files = st.multiselect('Files to Query', file_names[0]['file_names'])

                        
        

    #webbrowser.open('file://' + os.path.realpath(file_name))


    # the right main section
    st.header("Chat with Multiple Documents 🤖")

    # Capture User's prompt
    with st.form("prompt form", clear_on_submit=True):
        user_input = st.text_input("Ask a question about your documents: ", key="user_input", 
                    placeholder="Can you give me a short summary?", 
                   )
        st.form_submit_button("Enter", use_container_width=True)

    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant. ")
        ]
    
    # Manage context (memory)

    # generate GPT's response 
    if user_input:
        prompt = HumanMessage(content=user_input)
        st.session_state.messages.append(prompt)
        # clears input after user enters prompt
        with st.spinner("Thinking..."):
            # with open('file_names1.json', 'r',encoding='utf-8') as f:
            #     file_names = json.load(f)
            #     files = file_names[0]['file_names']
                search_output = model_response(user_input, selected_files)
                response =  search_output[0]
                sources = search_output[1]
                if ("Sorry" in response):
                    st.session_state.messages.append(AIMessage(content=str(response)))
                elif len(sources)>1:
                    st.session_state.messages.append(AIMessage(content=str(response)+" Sources: " + str(sources)))
                else: 
                    st.session_state.messages.append(AIMessage(content=str(response)+" Source: " + str(sources)))

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