from dotenv import load_dotenv
import os
from os.path import expanduser
from PyPDF2 import PdfReader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
#from langchain_mistralai.chat_models import ChatMistralAI
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chains.conversation.prompt import PROMPT
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, LlamaCppEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import glob

from langchain.llms import LlamaCpp

def get_pdf_text(pdf_docs):
    print("Getting texts from PDF")
    pdf_docs = glob.glob(expanduser("~/Downloads/*Tnr_ 37.pdf"))
    print(pdf_docs)
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    print("Chunking texts")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, embeddings):
    #FAISS.save_local("faiss_index")
    #new_db = FAISS.load_local("faiss_index", embeddings)
    print("FAISS Vectorstore")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_chroma_vectorstore(text_chunks, embeddings):
    print("Chroma Vectorstore")
    # save to disk
    #Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")

    # load from disk
    # Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(llm, vectorstore):  
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def main():
    load_dotenv()
    print("Started")

    llm = None
    memory = None
    prompt = PROMPT
    temperature = 0.1
    embeddings = None

    if os.getenv("OPENAI_API_KEY") is not None and os.getenv("OPENAI_API_KEY") != "":
        print("Using OPENAI")
        llm = ChatOpenAI(temperature=temperature)
        embeddings = OpenAIEmbeddings()

    if os.getenv("HUGGINGFACEHUB_API_TOKEN") is not None and os.getenv("HUGGINGFACEHUB_API_TOKEN") != "":
        print("Using HuggingfaceHub")
        llm = HuggingFaceHub(repo_id=os.getenv("HUGGINGFACE_HUB_MODEL"), model_kwargs={"temperature":temperature, "max_length":512})
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

 #   if os.environ.get("MISTRAL_API_KEY") is not None and os.environ.get("MISTRAL_API_KEY") != "":
 #       print("Using Mistral")
 #       llm =  ChatMistralAI()

    if os.getenv("LLAMA_FILE") is not None and os.getenv("LLAMA_FILE") != "":
        model_path=expanduser(os.getenv("LLAMA_FILE"))
        llm = LlamaCpp(
            model_path = model_path,
            streaming = False,
            temperature=temperature,
            max_tokens=2000,
            n_ctx=2048,
            top_p=1,
        )
        embeddings = LlamaCppEmbeddings(
            model_path=expanduser(os.getenv("LLAMA_FILE")),
            n_ctx=2048,
        )


    if llm == None :
        print("Please provide an LLM")
        exit(1)
    
    chatWith = input("Chat 1) directly with llm or 2) via VectorDB generated from dir >")

    if chatWith == "1":
        memory_model = input("memory model 1) Buffer 2) Entity >")

        if memory_model == "1":
            print("Using conversation buffer")
            memory = ConversationBufferMemory()
        else:
            print("Using Conversation entity buffer")
            memory = ConversationEntityMemory(llm=llm)
            prompt = ENTITY_MEMORY_CONVERSATION_TEMPLATE

        conversation = ConversationChain(
            llm = llm,
            verbose = True,
            memory = memory,
            prompt = prompt
        )

        print("Started llm")

        while True:
            user_input = input("> ")
            ai_response = conversation.predict(input=user_input)
            print("\n Assistent:\n", ai_response, "\n")
    else:
        vectorstore = None
        print("Generating vectorDB")
        text  = get_pdf_text(None)
        chunks = get_text_chunks(text)
        chooseVectorDB = input("Chat 1) FAISS 2) Chroma >")
        if chooseVectorDB == "2":
            vectorstore = get_chroma_vectorstore(text_chunks=chunks, embeddings=embeddings)
        else:     
            vectorstore = get_vectorstore(text_chunks=chunks, embeddings=embeddings)
        conversationChain = get_conversation_chain(llm=llm, vectorstore=vectorstore)
        while True:
            user_input = input("> ")
            ai_response = conversationChain({'question':user_input})
            print("\n Assistent:\n", ai_response['chat_history'], "\n")



if __name__ == '__main__':
    main()