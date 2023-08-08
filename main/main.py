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
from langchain.llms import OpenAI
from langchain.agents import create_csv_agent
from langchain.agents.agent_types import AgentType

# from st_custom_components import st_audiorec
import speech_recognition as sr
import pyttsx3
import tempfile
from audio_recorder_streamlit import audio_recorder
from pygame import mixer

openai.api_key = 'leina'


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
        new_pdf.add_font('Arial', '', 'c:/windows/fonts/arial.ttf', uni=True)
        new_pdf.set_font("Arial", size=12)
        fi = open(file_name, "r")
        for x in fi:
            # x = x.encode('latin-1', 'ignore')
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
    if not file_name.lower().endswith(".csv"):
        if (file_name in os.listdir(os.curdir)) and (file_name.lower().endswith('.docx.pdf') or file_name.lower().endswith('.txt.pdf')):
            os.remove(file_name.replace(".pdf", ""))
        os.remove(file_name+'.json')
    os.remove(file_name)

def SpeakText(path, text):
    #initialize the engine
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate-50)
    voices = engine.getProperty('voices')
    # print(len(voices))
    engine.setProperty('voice', voices[0].id)
    # engine.say(text)
    
    engine.save_to_file(text, path)
    engine.runAndWait()

def SpeachtoText(source):
    # Initialize the recognizer
    r = sr.Recognizer()

    # Loop in case of errors
    try:
        with source as source:
            # prepare recognizer to receive input
            r.adjust_for_ambient_noise(source, duration=0.2)
            print("I'm listening")

            # listen for the user's inpit
            audio2 = r.listen(source)
            #using google to recognize audio
            MyText = r.recognize_google(audio2)
            print(MyText)
            return MyText
    except sr.RequestError as e:
        print("Could not request results: {0}".format(e))

    except sr.UnknownValueError:
        print("unknown error occurred")

def play_audio(audio_file_path):
    mixer.init()
    mixer.music.load(audio_file_path)
    mixer.music.play()
    
# main fn
def main():
    # load API Key
    init()
    # the left sidebar section
    llm = OpenAI(temperature=0)
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
                    if not file_name.lower().endswith(".csv"):
                        extract(file_name)

        #represent current files to query/delete
        files = os.listdir(os.curdir)
        json_file_names = [k for k in files if '.json' in k]
        csv_file_names = [l for l in files if '.csv' in l]
        remove_file_names = []
        for j in json_file_names:
            remove_file_names.append(j)
        for c in csv_file_names:
            remove_file_names.append(c)
        with st.container():
            st.write('Files to remove')
            colrem1, colrem2 = st.columns([3,1.3])
            with colrem1:
                delete_files = st.multiselect('Files to Remove', remove_file_names, label_visibility="collapsed")
            with colrem2:
                if st.button('Remove', use_container_width=True):
                    for f in delete_files:
                        print(f)
                        delete(f.replace('.json', ''))
        tab1, tab2 = st.tabs(["text based queries", "data based queries"])

        with tab2:
            selected_csv= st.selectbox('CSV to Query', csv_file_names, key = "data")
            if selected_csv:
                agent = create_csv_agent(
                    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
                    # llm,
                    selected_csv,
                    verbose=True,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    return_intermediate_steps = True
                )
            usecsv = st.checkbox('Ask CSV files only')
        with tab1:
            selected_files = st.multiselect('Files to Query', json_file_names, disabled=usecsv, key = "selected")
            print(selected_files)
            usegpt = st.checkbox('Ask GPT')


    # the right main section
    st.header("Chat with Multiple Documents 🤖")

    user_input = ''

    usespeech = st.checkbox('Use Speech to Text')
    if usespeech:

        # #different UI
        # wav_audio_data = st_audiorec()

        wav_audio_data = audio_recorder(
            pause_threshold = 1.5,
            text="Click to ask your question",
            neutral_color="#FFFFFF",
            icon_size = '2x'
        )

        if wav_audio_data is not None:
            # # display audio data as received on the backend
            # recording = st.audio(wav_audio_data, format='audio/wav')

            with open('tempaudio1.wav', 'wb') as file:
                file.write(wav_audio_data)
            recording = sr.AudioFile('./tempaudio1.wav')
            print(recording)
            text = SpeachtoText(recording)
            user_input = text
    else:
        # #Capture User's prompt from chat box
        user_input = st.chat_input("Ask a question about your documents ")

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
                if usecsv:
                    response_dict = agent({"input": user_input})
                    response = response_dict['output']
                    steps = response_dict["intermediate_steps"]
                    if steps!=[]:
                        i1 = str(steps[0][0]).find('Invoking')
                        i2 = str(steps[0][0]).find('\\n', i1)
                        process = str(steps[0][0])[i1:i2]
                        response = "Steps to answer:\n" + process + "\n\n" + "Final answer: \n" + response
                else:
                    search_output = model_response(user_input, selected_files, usegpt)
                    response =  search_output[0]
                    sources = search_output[1]
                    if usespeech:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                                audio_file_path = f.name
                        try:
                            SpeakText(audio_file_path, response)
                            play_audio(audio_file_path)
                            # os.remove(audio_file_path)
                        except Exception as e:
                            st.error(f"Error: Failed to generate or play audio - {e}")

                if (("Sorry" in response) or usegpt or usecsv):
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