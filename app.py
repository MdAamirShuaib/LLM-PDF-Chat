import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub


def getPdfText(pdfDocs):
    text = ""
    for pdf in pdfDocs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def getTextChunks(rawText):
    splitter = CharacterTextSplitter(separator='\n',chunk_size=1000, chunk_overlap=200, length_function=len)
    textChunks = splitter.split_text(rawText)
    return textChunks

def getVectorStore(textChunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorStore = FAISS.from_texts(texts=textChunks, embedding=embeddings)
    return vectorStore

def getConversationChain(vectorStore):
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    llm=ChatOpenAI()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorStore.as_retriever(),
        memory=memory,
    )
    
    return chain



def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Chatbot", page_icon=":books:")
    st.header("PDF ChatBot :books:")
    st.text_input("Ask a question about your documents")

    with st.sidebar:
        st.subheader('Your Documents')
        pdfDocs = st.file_uploader("Upload your PDFs abd click on Process", type=['pdf'], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing your documents"):
                rawText = getPdfText(pdfDocs)
                textChunks = getTextChunks(rawText)
                vectorStore = getVectorStore(textChunks) 
                st.write(textChunks)
                conversation = getConversationChain(vectorStore)




if __name__ =='__main__':
    main()