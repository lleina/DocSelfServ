# DocSelfServ

The purpose of this project is to create a way to learn about a document without having to read through every page. 
The secondary goal is to be able to translate documents (takes in pdf, docx, and txt and outputs as docx or pdf).
To get started:
  1. Make sure you have Python3.11
  2. create a virtual environment
  3. create a requirements.txt by doing [pip install pipreqs]
  4. run [pip install -r requirements.txt]
    may require additional pip installs including:
        [pip install matplotlib]
        [pip install plotly]
        [pip install scipy]
        [pip install scikit-learn]
        [pip install speechrecognition]
        [pip install tabulate]
        [sudo apt install ffmpeg]
  5. Create a .env folder with OPENAI_API_KEY='your_openai_key' inside of it
  To access main application, go to main directory
  then run [streamlit run main.py]

  To access translation application, go to main>translation
  then run [streamlit run translation1.py]

  Features of main.py:
    -Manage a repository of pdf, docx, txt, csv
    -Chat with selected document(s)
    -Supports selected page ranges
    -Response includes sources
    -Ask GPT checkbox
    -Supports STT and TTS
  Features of translation1.py:
    -Generate translations of multi-paged documents
    -translate to english, spanish, chinese, german, french, italian, hindi, korean
    
  Note: speech may or may not work depending on OS. Translation works on Windows primarily if contain path to Arial font is c:/windows/fonts/arial.ttf. 
    

  
        
