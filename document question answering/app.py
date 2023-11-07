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

def main():
    load_dotenv()
    st.set_page_config(page_title= "Chat with multiple document(pdf)", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation= None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history= None

    st.header("Chat with docs... :books:")
    user_question= st.text_input("Ask question about your documents:")
    if user_question:
        handle_user_input(user_question)

    st.write(user_template.replace("{{MSG}}", "Hello Bot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)

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