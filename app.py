from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from fastapi import FastAPI, HTTPException, File, UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_chroma import Chroma
import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel
import streamlit as st
import tempfile
import requests
import uuid
import os

app = FastAPI(
    title="Chatbot API",
    description="API para um chatbot que responde perguntas com base no contexto passado (PDF)",
    version="0.1",
    docs_url="/",
)

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
newdb = Chroma(embedding_function=embeddings, persist_directory="data")

class RequestQuestion(BaseModel):
    question: str

@app.post("/postPDF")
async def readPDFConvert2Text(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="O arquivo precisa ser um PDF")

    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(contents)
            temp_pdf_path = temp_pdf.name

        pdf_reader = PyPDFLoader(temp_pdf_path)
        documents = pdf_reader.load_and_split()
        newdb.add_documents(documents)
        os.remove(temp_pdf_path)
        return {"message": "Arquivo carregado!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def save_interaction(question: str, answer: str):
    try:
        interaction_document = Document(
            page_content=answer,
            metadata={
                "id": str(uuid.uuid4()),
                "question": question,
                "answer": answer
            }
        )
        newdb.add_documents([interaction_document])
    except Exception as e:
        print(f"Erro ao salvar a interação: {e}")

@app.post("/askQuestion")
async def askQuestion(request: RequestQuestion):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        prompt_template = PromptTemplate.from_template(
            "Com base no contexto: {context}, responda a pergunta: {input}"
        )
        documents_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval = create_retrieval_chain(newdb.as_retriever(), documents_chain)
        response = retrieval.invoke({"input": request.question})
        answer = response.get('answer', response)

        save_interaction(request.question, answer)
        return {"answer": answer}
    except Exception as e:
        print(f"Erro: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def run_streamlit():
    st.set_page_config(layout="wide", page_title="PDF Q&A Chatbot")
    st.sidebar.title("Upload PDF")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload a PDF and ask me a question."}
        ]

    uploaded_file = st.sidebar.file_uploader("Escolha um arquivo PDF", type="pdf")

    if st.sidebar.button("Carregar PDF"):
        if uploaded_file is not None:
            with st.spinner("Carregando PDF..."):
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                response = requests.post(f"http://localhost:8001/postPDF", files=files)

                if response.status_code == 200:
                    st.sidebar.success("Arquivo carregado com sucesso!")
                else:
                    st.sidebar.error(f"Erro: {response.json().get('detail')}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_input := st.chat_input("Faça uma pergunta sobre o PDF:"):
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Buscando resposta..."):
                response = requests.post(
                    f"http://localhost:8001/askQuestion", json={"question": user_input}
                )
                
                if response.status_code == 200:
                    answer = response.json().get("answer")
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    st.write(answer)
                else:
                    st.error(f"Erro: {response.json().get('detail')}")

if __name__ == "__main__":
    import threading
    import uvicorn

    threading.Thread(target=uvicorn.run, args=(app,), kwargs={'host': '0.0.0.0', 'port': 8001}).start()

    run_streamlit()
