# Chatbot API

Esta é uma API desenvolvida com FastAPI para criar um chatbot que responde a perguntas com base no contexto de documentos PDF carregados. O chatbot utiliza o Google Generative AI e o Langchain para gerar respostas baseadas em documentos PDF fornecidos pelos usuários.

### OBSERVAÇÃO:
- Nesta Branche está o código pedido no desafio, mas adicionado uma interface usando Streamlit para melhor usabilidade e teste. 

## Funcionalidades

- Upload de arquivos PDF.
- Indexação de documentos PDF para recuperação eficiente de informações.
- Capacidade de responder a perguntas com base no conteúdo dos PDFs carregados.
- Salvamento de interações (perguntas e respostas) no banco de dados vetorial Chroma (inacabado).

## Tecnologias Utilizadas

- **[Streamlit](https://docs.streamlit.io/)**: Ferramenta para criar aplicações web interativas e fáceis de usar.
- **[FastAPI](https://fastapi.tiangolo.com/)**: Framework para construção de APIs.
- **[Langchain](https://langchain.com/)**: Para manipulação de modelos de linguagem e fluxos de trabalho.
- **[Google Generative AI](https://developers.generativeai.google/)**: Para geração de respostas e embeddings.
- **[Chroma](https://docs.trychroma.com/)**: Base de dados vetorial para armazenamento e recuperação de documentos.
- **[PyPDFLoader](https://python.langchain.com/docs/modules/data_connection/document_loaders/integrations/pdf)**: Para carregar e processar arquivos PDF.

## Requisitos

- Python 3.9 ou superior
- Instalar as dependências listadas no arquivo `requirements.txt`.

### Dependências

Aqui estão algumas das principais bibliotecas que precisam ser instaladas:

```bash
pip install langchain~=0.3.1 langchain_google_genai langchain-community langchain-chroma fastapi~=0.115.0 google-generativeai pypdf2 chromadb uvicorn protobuf~=4.25.5 python-multipart~=0.0.5 python-dotenv~=1.0.1 streamlit streamlit-chat
```

## Configuração

**1. Clone o repositório e utilize a Branche master**
```bash
git clone https://github.com/GabaDev2412/desafioLizardGenAIGeneration
```

**2. Crie um arquivo .env na raiz do projeto e adicione sua chave de API do Google Generative AI:**
```bash
GOOGLE_API_KEY="SUA CHAVE API"
```

**3. Instale as dependências:**
```bash
pip install -r requirements.txt
```

**4. Inicie o serviço Streamlit::**
```bash
streamlit run app.py
```

## Endpoints

### 1. Upload de PDF
`POST /postPDF`

Este endpoint permite o upload de um arquivo PDF. O conteúdo do PDF é indexado para uso futuro nas respostas.

#### Parâmetros:
- `file`: Arquivo PDF enviado pelo usuário.

#### Resposta:
- **Sucesso**: Retorna um JSON com a mensagem `"Arquivo carregado!"`.
- **Erro**:
  - Status **400** se o arquivo não for um PDF válido.
  - Status **500** se houver falha no processamento.

### 2. Fazer uma pergunta
`POST /askQuestion`

Este endpoint permite fazer perguntas com base no conteúdo do(s) PDF(s) carregado(s). A resposta será extraída e gerada a partir do contexto dos PDFs.

#### Parâmetros:
- `question`: Pergunta a ser respondida.

#### Exemplo de Corpo da Requisição:
```json
{
  "question": "Qual é o resumo do capítulo 2?"
}
```

### Resposta:
- **Sucesso**: Retorna um JSON com a resposta baseada no conteúdo do PDF.
- **Erro**:
  - Status **500** se houver falha no processamento da pergunta ou da resposta.

### Estrutura do Projeto
```bash
├── main.py                 # Arquivo principal da API
├── requirements.txt        # Dependências do projeto
├── README.md               # Documentação do projeto
├── .env                    # Arquivo de configuração de variáveis de ambiente
└── data/                   # Diretório persistente onde os embeddings vetoriais são armazenados
```