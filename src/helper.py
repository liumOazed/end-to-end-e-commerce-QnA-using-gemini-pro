from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA


load_dotenv()
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


llm = GoogleGenerativeAI(google_api_key=os.environ["GOOGLE_API_KEY"], 
                         model="gemini-flash",temperature=0.5)
# Create embeddings

instructor_embeddings = HuggingFaceInstructEmbeddings()
vector_db_file_path = "faiss_index"

def vector_db():
    # Load CSV File in memory using Langchain 
    loader = CSVLoader(file_path="../new_train.csv", source_column="question")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local("vectordb_file_path")#save it to a local 
    
def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vector_db_file_path, instructor_embeddings)
    
    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever()
    
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I can't assist you with that. For more info about the matter kindly talk to our Customer care service." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]) 

    chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff", 
            retriever=retriever, 
            input_key="query", return_source_documents=True,
            chain_type_kwargs = {"prompt": PROMPT})
    
    return chain
    