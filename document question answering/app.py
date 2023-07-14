import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory 
from langchain.chains import ConversationalRetrievalChain


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
    embeddings= HuggingFaceEmbeddings()
    # embeddings= HuggingFaceInstructEmbeddings(model_name= "hkunlp/instructor-xl")
    vectorstore= FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm= "xyz"
    memory= ConversationBufferMemory(memory_key='chat_history', return_messages= True)
    conversation_chain= ConversationalRetrievalChain.from_llm(
        llm= llm,
        retriever=vectorstore.as_retriever(),
        memory= memory
    )
    return conversation_chain



def main():
    load_dotenv()
    st.set_page_config(page_title= "Chat with multiple document(pdf)", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation= None

    st.header("Chat with docs... :books:")
    st.text_input("Ask question about your documents:")

    with st.sidebar:
        st.subheader("Your document here...")
        pdf_doc= st.file_uploader("Upload your pdf here...", accept_multiple_files= True)
        if st.button("Process"):
            with st.spinner("Processing"):

                #get the pdf text
                raw_text= get_pdf_text(pdf_doc)
                # st.write(raw_text)

                #split the text into chunks
                text_chunks= get_text_chunk(raw_text)


                #create the vectore store with embeddings
                vectorstore= get_vectorstore(text_chunks)
                st.write(vectorstore)


                #creating conversation chain
                st.session_state.conversation= get_conversation_chain(vectorstore)
    

if __name__ == '__main__':
    main()