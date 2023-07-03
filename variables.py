from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

num_files = 0
token_num = 4096
db = Chroma(persist_directory="./chroma_db", collection_name="uploaded_files", embedding_function=OpenAIEmbeddings())