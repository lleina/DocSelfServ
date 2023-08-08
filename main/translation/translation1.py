import streamlit as st
import os, uuid, PyPDF2, json, openai, docx2pdf
from dotenv import load_dotenv
from langchain.schema import (
    SystemMessage,
)
from fpdf import FPDF
from openai_multi_client import OpenAIMultiOrderedClient


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

def create_obj(content , file_name):
    obj = {
        "id": str(uuid.uuid4()),
        "text": content,
        "file_name": file_name,
    }
    return obj

def splitter(content, chunk_size):
    """Content is 1 page of information as a string. 
    Returns a list of chunks of the page. Each chunk is chunk_size characters +/- 200 characters.
    if true, need to append to next chunk. """

    total_char = len(content)
    start_idx = 0
    end_idx = start_idx + chunk_size
    chunks = [False, []]

    if end_idx>=total_char:
        chunks[1].append(content[start_idx:])

    while end_idx<total_char:
        if end_idx+200+1<total_char:
            found = False
            for c in ['.', ';', '!', '?']:
                if found == False:
                    k = content[start_idx:end_idx+200].rfind(c) + start_idx
                    if k!=-1 and content[start_idx:k+1]!='':
                        chunks[1].append(content[start_idx:k+1])
                        start_idx = k+1
                        found = True
            if (found == False) and content[start_idx:end_idx+1]!='':
                chunks[1].append(content[start_idx:end_idx+1])
                start_idx = end_idx+1

        else:
            chunks[1].append(content[start_idx:])
            start_idx = end_idx+1
            if (end_idx+200+1>=total_char):
                chunks[0]=True
        end_idx = end_idx+chunk_size


        
    return chunks

def pdf_helper(file_name, content_chunks, chunk_size):
    """chunk_size is the number of characters per each chunk"""
    pdf_file = open(file_name, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    # page_number = 1
    content_prev = ''
    for page in pdf_reader.pages:
        content = content_prev+' '+ page.extract_text()
        split = splitter(content, chunk_size)
        if split[0]==True:
            content_prev=split[1][len(split[1])-1]
        else:
            content_prev = ''
        for chunk in split[1]:
            obj = create_obj(chunk, file_name)
            # page_number +=1
            content_chunks.append(obj)
    pdf_file.close()
    return content_chunks

def extract(file_name, chunk_size):
    """returns a retriever for the given file_name
    uploaded_file: .pdf
    chunk_size is the number of characters per each chunk"""
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
        content_chunks = pdf_helper(file_name, content_chunks, chunk_size)
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

def translate_helper(api, dat, lang_option):
    print(type(lang_option))
    for item in dat:
        text_in = (item['text'])
        api.request(data={
            "messages": [{
                "role": "user",
                "content": "Translate the TEXT into {}. Here is the TEXT: {}".format(lang_option, text_in)
            }]
        }, metadata={'text': text_in})

def translate(file_name, lang_option):
    """req:
    file_names: .json and must exist in the same directory as main.py"""
    text_out = ''
    with open(file_name+'.json', 'r',encoding="utf-8") as jsonfile:
        dat = json.load(jsonfile)
        req = len(dat)

        api = OpenAIMultiOrderedClient(
        endpoint="chats",
        concurrency = req,
        data_template={"model": "gpt-3.5-turbo"}
        )
        api.run_request_function(translate_helper, api, dat, lang_option)

    for result in api:
        # text_in = result.metadata['text']
        response = result.response['choices'][0]['message']['content']
        text_out = text_out+ " " + response
    print(text_out)
    return text_out


def delete(file_name):
    """file should be the same name as the uploaded file"""
    os.remove(file_name)
    os.remove(file_name+'.json')

def create_pdf(file_name, text, lang_option):
    """Creates a pdf where each line is ~90 characters long, Arial size 11 font, left align"""
    new_pdf = FPDF()
    new_pdf.add_page()
    if (lang_option == "Chinese (Simplified)") or (lang_option == "Hindi"):
        new_pdf.add_font('unifont', '', './unifont-15.0.06.ttf', uni=True)
        new_pdf.set_font("unifont", size=11)
        line_size = 45
        print("CHINESE/HINDI DOC!!!")
    elif lang_option== "Korean":
        new_pdf.add_font('malgun', '', './MALGUN.TTF', uni=True)
        new_pdf.set_font("malgun", size=11)
        line_size = 55
        print("KOREAN DOC!!!")
    else: #english/spanish
        new_pdf.add_font('Arial', '', 'c:/windows/fonts/arial.ttf', uni=True)
        new_pdf.set_font("Arial", size=11)
        line_size = 95
        print("ENGLISH/LATIN/FRENCH/GERMAN")
    x=0
    while x<len(text):
        if x+line_size<len(text):
            s = text[x:x+line_size].rfind(' ')
            y = text[x:x+line_size].find('\n')
            if y!=-1:
                s = y
            if s == -1:
                new_pdf.cell(200, 10, txt=text[x:x+line_size], ln = 1, align = 'L')
                x = x+line_size+1
            else:
                new_pdf.cell(200, 10, txt=text[x:x+s], ln = 1, align = 'L')
                x= x+s+1
        else:
            new_pdf.cell(200, 10, txt=text[x:], ln = 1, align = 'L')
            x=x+line_size+1
    file_name = file_name + lang_option + ".pdf"
    files = os.listdir(os.curdir)
    if file_name in files:
        file_name = file_name + lang_option + 'new' + ".pdf"
    new_pdf.output(file_name)
    return file_name

# main fn
def main():
    # load API Key
    init()

    st.title("Translate Multiple Documents 🤖")
    # Upload pdf box and display upload document on screen
    uploaded_file = st.file_uploader("Upload your files and click on 'Process'", 
                                        accept_multiple_files = True)
    if uploaded_file !=None:
        # create select language
        lang_option = st.selectbox(
            "Translate to:",
            ("English", "Spanish", "Chinese (Simplified)", "French", "German", "Korean", "Hindi")
        )

        #create 'Process' button
        if st.button("Translate", use_container_width=True):
            for f in uploaded_file:
                file_name = f.name
                files = os.listdir(os.curdir)
                #handle update
                if file_name in files:
                    delete(file_name)
                save_uploaded_file(f)
                if not file_name.lower().endswith(".csv"):
                    chunk_size = 5000
                    extract(file_name, chunk_size)
                    t = translate(file_name, lang_option)
                    print(t)
                    trans_file_name = create_pdf(file_name, t, lang_option)
                    with open(trans_file_name, "rb") as fi:
                        st.download_button(
                            label = "download: " + trans_file_name,
                            data = fi,
                            file_name = trans_file_name,
                            mime = "application/pdf"
                        )

    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant. ")
        ]
    
    # generate GPT's response 


if __name__ == '__main__':
    main()