import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template, footer

# from langchain.llms import HuggingFaceHub
# from langchain.embeddings import HuggingFaceInstructEmbeddings


def getPdfText(pdfDocs):
    text = ""
    for pdf in pdfDocs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text


def getTextChunks(rawText):
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    textChunks = splitter.split_text(rawText)
    return textChunks


def getVectorStore(textChunks, selectedLLM, apiKey):
    if selectedLLM == "OpenAI":
        embeddings = OpenAIEmbeddings(openai_api_key=apiKey)
    # elif selectedLLM == "HuggingFace":
    #     embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorStore = FAISS.from_texts(texts=textChunks, embedding=embeddings)
    return vectorStore


def getConversationChain(vectorStore, selectedLLM, apiKey):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if selectedLLM == "OpenAI":
        llm = ChatOpenAI(openai_api_key=apiKey)
    # elif selectedLLM == "HuggingFace":
    #     llm = HuggingFaceHub(
    #         repo_id="google/flan-t5-xxl",
    #         model_kwargs={"temperature": 0.5, "max_length": 512},
    #         huggingfacehub_api_token=apiKey,
    #     )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorStore.as_retriever(),
        memory=memory,
    )
    return chain


def handleUserInput(userQuestion):
    if st.session_state.conversation:
        response = st.session_state.conversation({"question": userQuestion})
        st.session_state.chatHistory = response["chat_history"]
        for i, message in enumerate(st.session_state.chatHistory):
            if i % 2 == 0:
                st.write(
                    user_template.replace("{{MSG}}", message.content),
                    unsafe_allow_html=True,
                )
            else:
                st.write(
                    bot_template.replace("{{MSG}}", message.content),
                    unsafe_allow_html=True,
                )
    else:
        st.write("Please Choose a LLM and upload your documents to start chatting")


def main():
    apiConfiguration = False
    st.set_page_config(page_title="PDF Chatbot", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None

    st.header("PDF ChatBot :books:")
    userQuestion = st.text_input("Ask a question about your documents")
    if userQuestion:
        handleUserInput(userQuestion)

    with st.sidebar:
        if not apiConfiguration:
            st.header("API Configuration")
            # Removing the functionality for Hugging face as the free cloud platform doesnt have enough compute

            # options = ["Select a LLM", "OpenAI", "HuggingFace"]
            # default_option = "Select a LLM"

            # selectedLLM = st.selectbox(
            #     "Choose a LLM", options, index=options.index(default_option)
            # )
            selectedLLM = "OpenAI"
            if selectedLLM == "OpenAI":
                apiKey = st.text_input(
                    label="Enter your OpenAI API key (Paid and Fast)", type="password"
                )
                setKey = st.button("Set API key")
                if setKey:
                    os.environ["OPENAI_API_KEY"] = apiKey
                    apiConfiguration = True
                    st.write("API key set")

            # if selectedLLM == "HuggingFace":
            #     apiKey = st.text_input(
            #         label="HuggingFace API key (Free but Slow)", type="password"
            #     )
            #     setKey = st.button("Set API key")
            #     if setKey:
            #         os.environ["HUGGINGFACEHUB_API_TOKEN"] = apiKey
            #         apiConfiguration = True
            #         st.write("API key set")
            st.subheader("Your Documents")
            pdfDocs = st.file_uploader(
                "Upload your PDFs abd click on Process",
                type=["pdf"],
                accept_multiple_files=True,
            )
            if st.button("Process"):
                with st.spinner("Processing your documents"):
                    rawText = getPdfText(pdfDocs)
                    textChunks = getTextChunks(rawText)
                    vectorStore = getVectorStore(textChunks, selectedLLM, apiKey)
                    st.session_state.conversation = getConversationChain(
                        vectorStore, selectedLLM, apiKey
                    )
                st.write("Documents Processed")
    st.markdown(footer, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
