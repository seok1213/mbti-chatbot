from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI 임베딩 초기화
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 텍스트 로더 초기화
loader = TextLoader("mbti_texts/INFP.txt", encoding="utf-8")
data = loader.load()

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(data)

# Chroma 데이터베이스 생성
persist_directory = "./chromadb/infp"
db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)

# 데이터베이스 내 문서 개수 확인
print(f"Number of documents in Chroma DB: {len(db.get())}")

# Retriever 생성 및 유사도 점수 임계값 설정
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.76}  # 임계값을 낮춰 테스트
)

# 질의
query = "안녕?"

# 검색 결과 확인
try:
    relevant_docs = retriever.get_relevant_documents(query)
    if not relevant_docs:
        print("No relevant documents found.")
    else:
        print("Relevant documents:")
        for doc in relevant_docs:
            print(doc.page_content)
            print("=========================================================")
except Exception as e:
    print(f"An error occurred: {e}")




