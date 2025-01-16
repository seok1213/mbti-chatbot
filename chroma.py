import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv


from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

mbti = 'estp'

def load_and_split_txt(txt_path, chunk_size=100, chunk_overlap=0):
    # TXT 파일 로드
    loader = TextLoader(txt_path, encoding="utf-8")
    data = loader.load()

    # 텍스트 분할기 초기화
    text_splitter = RecursiveCharacterTextSplitter(
        separators = '\n',
        chunk_size = chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    # 텍스트 분할 후 반환
    return text_splitter.split_documents(data)


texts = load_and_split_txt(f'data/{mbti}.txt')


# .env 파일 로드
load_dotenv()

# OpenAI API 키 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI 임베딩 초기화
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

db = Chroma.from_documents(texts, embeddings, persist_directory= f"./data/embedding/{mbti}_chroma_db")

