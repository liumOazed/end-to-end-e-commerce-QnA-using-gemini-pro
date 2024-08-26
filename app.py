import streamlit as st
from src.helper import vector_db, get_qa_chain
st.title("E-Commerce Customer QA")
btn = st.button("Create Knowledgebase")

if btn:
    pass

question = st.text_input("Enter Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)
    
    st.header("Answer: ")
    st.write(response["result"])