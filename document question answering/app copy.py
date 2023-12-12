import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory 
from langchain.chains import ConversationalRetrievalChain
from htmltemplate import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
# from langchain.llms import CTransformers
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os

def get_pdf_text(pdf_docs):
    #variable to store all of the text from pdf
    text= ""
    for pdf in pdf_docs:
        #initialize one pdfReader object for each pdf
        pdf_reader= PdfReader(pdf)
        #loop through every pages in pdf
        for page in pdf_reader.pages:
            #appending all the text into text variable
            text+=page.extract_text()
    return text

def get_text_chunk(raw_text):
    text_splitter= CharacterTextSplitter(separator="\n", chunk_size= 1000, chunk_overlap= 200, length_function= len)
    chunks= text_splitter.split_text(raw_text)
    st.write(chunks)
    return chunks

def get_vectorstore(text_chunks):
    # embeddings= HuggingFaceEmbeddings()
    embeddings= OpenAIEmbeddings()
    # embeddings= HuggingFaceInstructEmbeddings(model_name= "hkunlp/instructor-xl")
    vectorstore= FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):

    #using Hugging face hub
    # llm= HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature":0.5, "max_length":1000})
    llm= ChatOpenAI(temperature=0.3, max_tokens=200)


    #using Ctransformer LLM....(library not working)
    # llm= CTransformers(model="marella/gpt-2-ggml", callbacks=[StreamingStdOutCallbackHandler()])

    memory= ConversationBufferMemory(memory_key='chat_history', return_messages= True)
    conversation_chain= ConversationalRetrievalChain.from_llm(
        llm= llm,
        retriever=vectorstore.as_retriever(),
        memory= memory
    )
    return conversation_chain


def handle_user_input(user_question):
    response= st.session_state.conversation({'question': user_question})
    # st.write(response)
    st.session_state.chat_history= response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# ... [previous imports and functions]

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple document(pdf)", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with docs... :books:")

    # User selects the directory
    input_dir = st.text_input("Enter the input directory path: ")
    if input_dir and os.path.isdir(input_dir):
        # List directories within the input directory
        folders = [folder for folder in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, folder))]
        selected_folder = st.selectbox("Select a folder:", folders)
        
        # List PDF files in the selected folder
        folder_path = os.path.join(input_dir, selected_folder)
        st.write("Selected folder path:", folder_path)  # Debugging line

        

        pdf_files = [file for file in os.listdir(folder_path) if file.endswith(".pdf")]
         # Debugging: Output the found PDF files
        st.write("PDF files found:", pdf_files)  # Debugging line

        if pdf_files:  # Check if there are PDF files in the folder
            selected_pdf = st.selectbox("Select a PDF file:", pdf_files)
            if st.button("Process"):
                with st.spinner("Processing"):
                    file_path = os.path.join(folder_path, selected_pdf)
                    pdf_docs = [file_path]
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunk(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
        else:
            st.write("No PDF files found in the selected folder.")

    user_question = st.text_input("Ask a question about the selected document")
    if user_question:
        handle_user_input(user_question)

if __name__ == '__main__':
    main()
