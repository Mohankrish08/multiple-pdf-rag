import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub

st.title("Custom LLM Model")
st.write("This app uses a pre-trained Mistral model to generate text.")

prompt = st.text_area("Enter a prompt:", height=100)

if st.button("Generate Text"):
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        model_kwargs={"temperature": 0.3, "max_length": 1080},
        huggingfacehub_api_token='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'
    )
    response = llm(prompt)
    st.write("Generated text:", response.text)
