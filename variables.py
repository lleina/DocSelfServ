from langchain.vectorstores import Chroma

num_files = 0
token_num = 4096
db = Chroma(persist_directory="./chroma_db")