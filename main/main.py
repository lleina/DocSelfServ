import streamlit as st
import os, uuid, PyPDF2, json, openai, docx2pdf
import numpy as np
from streamlit_chat import message
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from openai.embeddings_utils import get_embedding,cosine_similarity
from fpdf import FPDF
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
openai.api_key = ''

def init():
    """Sets API Key"""
    st.set_page_config(layout="wide")
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

def create_obj(content, page_number,file_name):
    obj = {
        "id": str(uuid.uuid4()),
        "text": content,
        "page": page_number,
        "file_name": file_name,
        "embedding": get_embedding(content,engine='text-embedding-ada-002')
    }
    return obj

def pdf_helper(file_name, content_chunks):
    pdf_file = open(file_name, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    page_number = 1
    for page in pdf_reader.pages:
        content = page.extract_text()
        obj = create_obj(content, page_number, file_name)
        page_number +=1
        content_chunks.append(obj)
    pdf_file.close()
    return content_chunks

def extract(file_name):
    """returns a retriever for the given file_name
    uploaded_file: .pdf"""
    content_chunks = []
    #for pdf
    if file_name.lower().endswith(".docx"):
        old_file_name = file_name
        file_name = file_name+".pdf"
        temp_file = open(file_name, "w")
        temp_file.close()
        docx2pdf.convert(old_file_name, file_name)
    elif file_name.lower().endswith(".txt"):
        new_pdf = FPDF()
        new_pdf.add_page()
        new_pdf.set_font("Arial", size=12)
        fi = open(file_name, "r")
        for x in fi:
            new_pdf.cell(200, 10, txt=x, ln = 1, align = 'L')
        file_name = file_name + ".pdf"
        new_pdf.output(file_name)

    if file_name.lower().endswith(".pdf"):
        content_chunks = pdf_helper(file_name, content_chunks)
    else:
        print("FILE TYPE IS NOT SUPPORTED! ONLY .PDF AND .DOCX")
        return None

    json_file_path = file_name+'.json'
    with open(json_file_path, 'a+',encoding='utf-8') as f:
        f.write("[ \n\n]")
    with open(json_file_path, 'r',encoding='utf-8') as f:
        data = json.load(f)
        
    for i in content_chunks:
            data.append(i)
    with open(json_file_path, 'w',encoding='utf-8') as f:
        json.dump(data, f,ensure_ascii=False, indent=4)
    

def model_response(user_input, file_names, usegpt):
    """req:
    file_names: .json and must exist in the same directory as main.py"""
    user_query_vector = get_embedding(user_input,engine='text-embedding-ada-002')
    unsorted_data = []
    for file in file_names:
        with open(file, 'r',encoding="utf-8") as jsonfile:
            data = json.load(jsonfile)
            for item in data:
                item['embeddings'] = np.array(item['embedding'])
            for item in data:
                item['similarities'] = cosine_similarity(item['embedding'], user_query_vector)
                unsorted_data.append(item)
    sorted_data = sorted(unsorted_data, key=lambda x: x['similarities'], reverse=True)
    context = ''
    # source = []
    source = '\n\n Sources: \n'
    for item in sorted_data[:3]:
        context += item['text']+";   "
        if (item['file_name'] + " page: " + str(item['page'])) not in source:
            # source.append(item['file_name']+", page: " + str(item['page']))
            source = source+'- '+ item['file_name']+", page: " + str(item['page']) + "\n"
    if context == '':
            context = 'There is NO CONTENT!'
    if usegpt:
        myMessages = [
            {"role": "system", "content": "You're a helpful Assistant."},
            {"role": "user", "content": "Answer the following QUERY:\n ### {} ###\n\n using the CONTENT:\n### {}### \n\n If the answer isn't found in the CONTENT provided, go ahead and create a response using other information and make sure to cite sources at the end of your response in the following format### \n Sources: \n-SOURCE \n -SOURCE \n-SOURCE ### where SOURCE is a variable you replace with the sources you used. If there is no source found, please say 'sources not found' ".format(user_input,context)}
        ]
    else: 
        myMessages = [
            {"role": "system", "content": "You're a helpful Assistant."},
            {"role": "user", "content": "Answer the following QUERY:\n ### {} ###\n\n using the CONTENT:\n### {}### \n\n If the answer isn't found in the CONTENT provided, always respond with exactly this sentence: Sorry, the content does not contain that information. ".format(user_input,context)}
        ]
    print({"role": "user", "content": "Answer the following QUERY:\n ### {} ###\n\n using the CONTENT:\n### {}### \n\n If the answer isn't found in the CONTENT provided, always respond with exactly this sentence: Sorry, the content does not contain that information. ".format(user_input,context)})
    
    # # To keep memory of conversation
    # conversation = ConversationChain(
    #     llm = ChatOpenAI(temperature=0),
    #     verbose = True,
    #     memory = ConversationBufferMemory()
    # )
    # response = conversation.predict(input=str(myMessages)) 
    # return [str(response), source]

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=myMessages,
        max_tokens=500,
    )
    #print(source)
    return [response['choices'][0]['message']['content'], source]

def delete(file_name):
    """file should be the same name as the uploaded file"""
    if (file_name in os.listdir(os.curdir)) and (file_name.lower().endswith('.docx.pdf') or file_name.lower().endswith('.txt.pdf')):
        os.remove(file_name.replace(".pdf", ""))
    os.remove(file_name+'.json')
    os.remove(file_name)

# main fn
def main():
    # load API Key
    init()
    # the left sidebar section

    with st.sidebar:
        st.title("Your documents")
        # Upload pdf box and display upload document on screen
        uploaded_file = st.file_uploader("Upload your files and click on 'Process'", 
                                         accept_multiple_files = True)
        if uploaded_file !=None:
            #create 'Process' button
            if st.button("Process", use_container_width=True):
                for f in uploaded_file:
                    file_name = f.name
                    files = os.listdir(os.curdir)
                    #handle update
                    if file_name in files:
                        delete(file_name)
                    save_uploaded_file(f)
                    extract(file_name)

        #represent current files to query/delete
        files = os.listdir(os.curdir)
        json_file_names = [k for k in files if '.json' in k]
        with st.container():
            st.write('Files to remove')
            
            colrem1, colrem2 = st.columns([3,1.3])
            with colrem1:
                delete_files = st.multiselect('Files to Remove', json_file_names, label_visibility="collapsed")
            with colrem2:
                if st.button('Remove', use_container_width=True):
                    for f in delete_files:
                        print(f)
                        delete(f.replace('.json', ''))
        selected_files = st.multiselect('Files to Query', json_file_names, key = "selected")
        print(selected_files)
        colcheck1, colcheck2 = st.columns([2,1.1])
        # with colcheck1:
            # query_all = st.checkbox('Query all documents')
            # if query_all:
            #     selected_files = st.multiselect('Files to Query', options = json_file_names, default = json_file_names, key = "selected2")
        with colcheck2:
            usegpt = st.checkbox('Ask GPT')

    # the right main section
    st.header("Chat with Multiple Documents 🤖")

    # #Capture User's prompt
    # with st.form("prompt form", clear_on_submit=True):
    user_input = st.chat_input("Ask a question about your documents ")
        # st.form_submit_button("Enter", use_container_width=True)

    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant. ")
        ]
    
    # generate GPT's response 
    if user_input:
        prompt = HumanMessage(content=user_input)
        st.session_state.messages.append(prompt)
        # clears input after user enters prompt
        with st.spinner("Thinking..."):
                search_output = model_response(user_input, selected_files, usegpt)
                response =  search_output[0]
                sources = search_output[1]
                if (("Sorry" in response) or usegpt):
                    st.session_state.messages.append(AIMessage(content=str(response)))
                else:
                    st.session_state.messages.append(AIMessage(content=str(response)+ str(sources)))
                
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