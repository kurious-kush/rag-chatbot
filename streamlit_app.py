import streamlit as st
import time
from groq import Groq  # Import Groq

# Set your Groq API key here
GROQ_API_KEY = "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"

# Import your existing code
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import cassio
from langchain.vectorstores.cassandra import Cassandra
from langchain.embeddings import GooglePalmEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from astrapy import DataAPIClient

# Set your Google API key here
GOOGLE_API_KEY = "AIzaSyBpZJYfrHuVqcWG5WaTXQ_-vOKMc3FIRPc"

# Provide Astra DB connection details
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:LfvQnAQHwfAThhucwFQsCfSf:9ce730d740d699dbb1213b5bf88a8680457263237f9e1c85dcf5ec54947c8c2d"
ASTRA_DB_ID = "86538be5-7d9d-4c2a-9843-e8560b179d01"

# Initialize the client
client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
db = client.get_database_by_api_endpoint(
  "https://86538be5-7d9d-4c2a-9843-e8560b179d01-us-east1.apps.astra.datastax.com"
)

print(f"Connected to Astra DB: {db.list_collection_names()}")


import fitz  # PyMuPDF

def process_pdf(pdf_path):
    text = ""
    try:
        # Open the PDF file
        with fitz.open(pdf_path) as doc:
            # Extract text from each page
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error processing PDF: {e}")
    return text

    pass

def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def add_pdf_to_db(pdf_docs):
    # Process PDF files and add text chunks to Astra DB
    for pdf_file in pdf_docs:
        raw_text = process_pdf(pdf_file)
        text_chunks = chunk_text(raw_text)
        # Add text chunks to Astra DB vector store
        cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
        astra_vector_store = Cassandra(
            embedding=GooglePalmEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY),
            table_name="your_table_name"
        )
        astra_vector_store.add_texts(text_chunks)

def user_input(question, google_api_key, model_type):
    if model_type == "Gemini":
        # Load Astra DB vector store
        cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
        astra_vector_store = Cassandra(
            embedding=GooglePalmEmbeddings(model="models/embedding-001", google_api_key=google_api_key),
            table_name="your_table_name"
        )

        # Search for similar documents
        docs = astra_vector_store.similarity_search(question)

        # Initialize conversational chain
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=google_api_key)
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        # Generate response
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        return response["output_text"]
    elif model_type == "Groq":
        # Initialize Groq model with your API key
        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": question,
                }
            ],
            model="llama3-70b-8192",
        )
        return chat_completion.choices[0].message.content

def main():
    # Set the page configuration as the first Streamlit command
    st.set_page_config(page_title="Chat PDF", layout="wide")

    # Custom CSS for background color and animation
    main_bg = "#f0f0f0"
    main_bg_color = f"background-color: {main_bg};"
    st.markdown(
        f"""
        
        """,
        unsafe_allow_html=True
    )

    st.header("AHOY!:) I am Gemini💁 lets take RAG to next level")
    user_question = st.text_input("Ask any Question from the PDF Files")

    # Add a selectbox for choosing the model type
    model_type = st.selectbox("Select Chat Model", ("Gemini", "Groq"))

    if user_question:
        st.write("You:", user_question)
        st.write(model_type + ":", "Thinking...")

        with st.spinner(model_type + " is thinking..."):
            time.sleep(3)  # Simulating processing time

            # Generate response
            response = user_input(user_question, GOOGLE_API_KEY, model_type)

            st.write(model_type + ":", response)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Add uploaded PDF files to Astra DB
                add_pdf_to_db(pdf_docs)
                st.success("PDFs added to Astra DB")

if __name__ == "__main__":
    main()
