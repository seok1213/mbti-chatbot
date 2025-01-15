import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# OpenAI API 키 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI 임베딩 초기화
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 텍스트 파일을 MBTI 유형별로 저장하는 함수
def embed_and_store_mbti_data():
    mbti_texts_dir = "mbti_texts"
    if not os.path.exists(mbti_texts_dir):
        print(f"'{mbti_texts_dir}' 디렉터리가 존재하지 않습니다.")
        return

    # MBTI 텍스트 디렉터리에서 .txt 파일만 가져오기
    files = [f for f in os.listdir(mbti_texts_dir) if f.endswith(".txt")]

    if not files:
        print(f"'{mbti_texts_dir}' 디렉터리에 텍스트 파일이 없습니다.")
        return

    for file_name in files:
        mbti_type = os.path.splitext(file_name)[0]  # 파일 이름에서 확장자 제거
        file_path = os.path.join(mbti_texts_dir, file_name)

        # 파일 읽기
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # 각 MBTI 유형별로 별도의 컬렉션 생성
        persist_directory = f"./chromadb/{mbti_type}"
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        # 텍스트 데이터 추가
        vectorstore.add_texts(
            texts=[content],
            metadatas=[{"mbti": mbti_type}],
            ids=[f"{mbti_type}_doc"]
        )
        print(f"{mbti_type} 데이터가 '{persist_directory}'에 임베딩되어 저장되었습니다.")

        # ChromaDB 저장
        vectorstore.persist()

# 데이터 임베딩 및 저장 실행
embed_and_store_mbti_data()





