from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from fastapi import FastAPI, HTTPException, File, UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import google.generativeai as genai
from pydantic import BaseModel
from dotenv import load_dotenv
import tempfile
import uuid
import os

# IMPORTAÇÕES QUE NÃO FORAM UTILIZADAS, MAS SE UTILIZÁSSEMOS O METODO DE CHUNKS DO PDF, SERIAM NECESSÁRIAS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_core.output_parsers import StrOutputParser
# from langchain.schema import Document
# from PyPDF2 import PdfReader

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

class requestQuestion(BaseModel):
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

        # OUTRA FORMA DE DIVIDIR O TEXTO EM CHUNKS E ATRIBUIR AOS DOCUMENTOS UM DICIONÁRIO DE METADADOS
        # pdf_reader = PdfReader(temp_pdf_path)
        # context = ""
        # for page in pdf_reader.pages:
        #     context += page.extract_text()
        #
        # # Dividindo o texto em chunks
        # splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=1000, chunk_overlap=1000
        # )
        #
        # chunks = splitter.split_text(context)
        # documents = [Document(page_content=chunk, metadata={"id": str(uuid.uuid4())}) for chunk in chunks]

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
        print(f"Documento criado: {interaction_document}")

        newdb.add_documents([interaction_document])
        print(f"Interação salva: {interaction_document.metadata}")
    except Exception as e:
        print(f"Erro ao salvar a interação: {e}")


@app.post("/askQuestion")
async def askQuestion(request: requestQuestion):
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

        print(f"Resposta completa do modelo: {answer}")
        return {"answer": answer}
    except Exception as e:
        print(f"Erro: {e}")
        raise HTTPException(status_code=500, detail=str(e))




