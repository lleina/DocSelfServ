import openai
import os
import streamlit as st
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from tempfile import NamedTemporaryFile

def main():

    openai.api_key = ""
    os.environ["OPENAI_API_KEY"] = ""

    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        with NamedTemporaryFile(mode='w+b', suffix=".csv") as f:
            f.write(csv_file.getvalue())
            f.flush()
            llm = OpenAI(temperature=0)
            user_input = st.chat_input("Enter your question here:")
            agent = create_csv_agent(llm, f.name, verbose=True)
            if user_input:
                with st.spinner("Working..."):
                    response = agent.run(user_input)
                    st.write(response)

if __name__ == "__main__":
    main()