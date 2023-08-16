import streamlit as st
import os, uuid, PyPDF2, json, openai, docx2pdf
from dotenv import load_dotenv
from fpdf import FPDF
from pdf2docx import Converter
from openai_multi_client import OpenAIMultiOrderedClient

# README
# GOALS: Translates documents into multiple languages using OpenAI Api Key.
# REQUIRES a .env folder with OPENAI_API_KEY=key inside of it. Also requires
# Several .ttf files including MALGUN.TTF, NotoSansDevangari-Regular.ttf, 
# unifont-15.0.06.ttf in the same folder as this python code.
# TO RUN: type >streamlit run translation1.py

openai.api_key = 'your_openai_key' #replace with your openai API key

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
    """
    uploaded_file : file type
    Saves the uploaded_file
    """
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
    """
    content : str
    chunk_size : int 
    Returns a list where the first element is a bool representing if the last chunk needs to be
    appended to the next string of content to be split.
    the second element is a list of strings representing chunks of the content.
    Each chunk is chunk_size characters +/- 200 characters.
    """
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
    """
    file_name : str representing a .pdf file in the directory
    content_chunks : str
    chunk_size : int
    Creates chunks of content from a .pdf file
    """
    pdf_file = open(file_name, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
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
            content_chunks.append(obj)
    pdf_file.close()
    return content_chunks

def extract(file_name, chunk_size):
    """
    file_name: str representing the name of either a .docx, .txt, or .pdf file in the directory
    chunk_size: int representing the characters in each chunk of text
    Creates a json file that contains a recrods of each string of text of size chunk_size.
    The records will include 'id', 'text', and 'file_name'
    """
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
        fi = open(file_name, "r", encoding="utf8")
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
    """
    api : OpenAIMultiOrderedClient
    dat : deserialized json like object
    lang_option: str
    """
    for item in dat:
        text_in = (item['text'])
        api.request(data={
            "messages": [{
                "role": "user",
                "content": "Translate the TEXT into {}. Here is the TEXT: {}".format(lang_option, text_in)
            }]
        }, metadata={'text': text_in})

def translate(file_name, lang_option):
    """
    file_name: str representing the name of the uploaded file.
    lang_option: str
    Requires file_name.json to exist in the same directory.
    Returns a str that is the translated version of all the text in file_name.json
    """
    text_out = ''
    with open(file_name+'.json', 'r',encoding="utf-8") as jsonfile:
        dat = json.load(jsonfile)
        req = len(dat)
        api = OpenAIMultiOrderedClient(
        endpoint="chats",
        concurrency = req, #var determines how many concurrent calls at once !!!IMPORTANT!!!!
        data_template={"model": "gpt-3.5-turbo"}
        )
        api.run_request_function(translate_helper, api, dat, lang_option)

    for result in api:
        response = result.response['choices'][0]['message']['content']
        text_out = text_out+ " " + response

    print(text_out)
    return text_out


def delete(file_name, lang_option):
    """
    file_name: str representing the name of the uploaded file
    lang_option: str that's in the list of languages for translation

    Deletes file_name and other files created by file_name in the case where
    there's another upload of the same file_name.
    """
    os.remove(file_name)
    trans_pdf = file_name_creator(file_name, lang_option, False)
    trans_docx = trans_pdf[:len(trans_pdf)-4]+'.docx'
    files = os.listdir(os.curdir)
    if trans_pdf in files:
        os.remove(trans_pdf)
    if trans_docx in files:
        os.remove(trans_docx)
    if file_name.lower().endswith(".pdf"):
        if file_name+'.json' in files:
            os.remove(file_name+'.json')
    else:
        if file_name+'.pdf'in files:
            os.remove(file_name+'.pdf')
        if file_name+'.pdf.json' in files:
            os.remove(file_name+'.pdf.json')

def file_name_creator(file_name, lang_option, exists):
    """pdf name creator for a translated file. 
    Returns file_name_langoption.pdf if exists is False, otherwise
    it returns file_name_langoption_new.pdf if exists is True"""
    if exists:
        return file_name + "_" + lang_option + 'new' + ".pdf"
    else:
        return file_name + "_" + lang_option + ".pdf"
    
def create_pdf_helper(new_pdf, font_name, ttf_file, font_size):
    """
    new_pdf : FPDF
    font_name : str
    ttf_file : str
    font_size: int
    """
    new_pdf.add_font(font_name, '', ttf_file, uni=True)
    new_pdf.set_font(font_name, size=font_size)

def create_pdf_helper2(lang_option, font_size):
    new_pdf = FPDF()
    new_pdf.add_page()
    if (lang_option == "Chinese (Simplified)") or (lang_option == "Hindi"):
        font_name = 'unifont'
        ttf_file = './unifont-15.0.06.ttf'
        create_pdf_helper(new_pdf, font_name, ttf_file, font_size)
        line_size = 45
    elif lang_option== "Korean":
        font_name = 'malgun'
        ttf_file = './MALGUN.TTF'
        create_pdf_helper(new_pdf, font_name, ttf_file, font_size)
        line_size = 55
    else: #english/spanish/french/german/italian
        font_name = 'Arial'
        ttf_file = 'c:/windows/fonts/arial.ttf' #if not using windows, may need to import Arial into folder
        create_pdf_helper(new_pdf, font_name, ttf_file, font_size)
        line_size = 95
    return [line_size, new_pdf]

def create_pdf(file_name, text, lang_option, font_size):
    """
    Params
    file_name: string of file name
    text: a string that will appear in output pdf
    lang_option: the language of the text. Has to be what's supported in the streamlit selectbox
    font_size: an int. 11 is recommended
    Requires ./unifont-15.0.06.ttf, ./MALGUN.TTF, in current directory and c:/windows/fonts/arial.ttf.
    --------------------------------------------------------------------------------------------------
    Creates a pdf with left align text. font and number of characters per line
    vary based on language. If desire different font, import the .ttf font into current folder and
    change font_name and ttf_file accordingly based on language.
    Returns a pdf file name generated by file_name_creator
    """
    new_pdf_created = create_pdf_helper2(lang_option, font_size)
    line_size = new_pdf_created[0]
    new_pdf = new_pdf_created[1]
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

    file_name =  file_name_creator(file_name, lang_option, False)
    files = os.listdir(os.curdir)
    if file_name in files:
        file_name = file_name_creator(file_name, lang_option, True)
    new_pdf.output(file_name)
    return file_name

def docx_creator(trans_file_name):
    """
    Requires trans_file_name : string name of a .pdf file in the same folder
    Creates a .docx file from a .pdf file
    """
    docx = Converter(trans_file_name)
    docx_name = trans_file_name[:len(trans_file_name)-4]+'.docx'
    docx.convert(docx_name, start=0, end=None)
    docx.close()
    return docx_name

# main fn
def main():
    # load API Key
    init()

    st.title("Translate Documents")
    # Upload pdf box and display upload document on screen
    uploaded_file = st.file_uploader("Upload your file and click on 'Translate'. Supports .pdf, .docx, .txt", 
                                        accept_multiple_files = True)
    if uploaded_file !=None:
        # create select language
        lang_option = st.selectbox(
            "Translate to:",
            ("English", "Spanish", "Chinese (Simplified)", "French", "German", "Korean", "Hindi", "Italian")
        )
        output_format = st.selectbox(
            "Output file format:",
            ("PDF", "DOCX")
        )

        #vars to create PDF
        chunk_size = 5000
        font_size = 11

        #create 'Translate' button
        if st.button("Translate", use_container_width=True):
            for f in uploaded_file:
                file_name = f.name
                files = os.listdir(os.curdir)
                #handles update
                if file_name in files:
                    delete(file_name, lang_option)
                save_uploaded_file(f)
                if not file_name.lower().endswith(".csv"):
                    extract(file_name, chunk_size)
                    if not file_name.lower().endswith(".pdf"):
                        t = translate(file_name+'.pdf', lang_option)
                    else:
                        t = translate(file_name, lang_option)
                    # print(t)
                    trans_file_name = create_pdf(file_name, t, lang_option, font_size)
                    if "DOCX" in output_format:
                        docx_name = docx_creator(trans_file_name)
                        #creates a download button for docx
                        with open(docx_name, "rb") as fi:
                            st.download_button(
                                label = "download: " + docx_name,
                                data = fi,
                                file_name = docx_name,
                                mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
                    if "PDF" in output_format:
                        #creates a download button for pdf
                        with open(trans_file_name, "rb") as fi:
                            st.download_button(
                                label = "download: " + trans_file_name,
                                data = fi,
                                file_name = trans_file_name,
                                mime = "application/pdf"
                            )
                else:
                    st.write("Doesn't translate .csv")

if __name__ == '__main__':
    main()