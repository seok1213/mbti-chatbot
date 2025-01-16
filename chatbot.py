from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from utils import load_and_split_txt, format_docs
from config import EMBEDDING_DB_PATH, MBTI_FEATURES, TXT_PATHS
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os


# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class MBTIChatBot:
    def __init__(self, mbti, temperature=0):
        self.mbti = mbti
        self.feature = MBTI_FEATURES[mbti]
        self.history_path = f"data/{mbti}/{mbti}_history.txt"  # 대화 기록 파일 경로
        self.txt_path = TXT_PATHS[mbti]  # TXT 파일 경로를 가져옵니다.
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.db = Chroma(
            persist_directory=f"{EMBEDDING_DB_PATH}{mbti}_chroma_db",
            embedding_function=self.embeddings
        )
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature, openai_api_key=OPENAI_API_KEY)
        self.chat_template = ChatPromptTemplate.from_template(
            """
    [MBTI 정보]
    - MBTI 유형: {mbti}
    - 주요 특징: {feature}

    [대화 예시]
    {context}



    [현재 질문]
    Q: {query}

    [지시사항]
    1. 너는 {mbti} 유형의 사람처럼 행동해야 해.
    2. {feature}와 [대화 예시], [이전 대화 기록]을 참고하여 대답을 만들어야 해.
    3. 답변은 {mbti}의 특징을 반영해 자연스럽고 친근한 톤으로 작성해.
    4. 대답은 가볍고 짧게, 직관적으로 만들어야 해.
    5. 사용자는 너와 같은 MBTI 유형이 아니며, 이를 고려해 적절한 답변을 작성해.
    6. 질문이 추상적이거나 개인적인 경우, {mbti} 유형이 일반적으로 선호하는 답변 방향으로 답해.
    7. 이전 대화를 바탕으로 대답할 수 있으면 그렇게 해줘.

    [답변]
    A:
             """
        )

    def initialize_db(self):
        docs = load_and_split_txt(self.txt_path)  # TXT 파일을 로드하고 분할합니다.
        self.db.add_documents(docs)

    def get_response(self, query):
        retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.6}
        )
        relevant_docs = retriever.get_relevant_documents(query)

        unique_docs = []
        seen_contents = set()

        for doc in relevant_docs:
            if doc.page_content not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(doc.page_content)
         
        
        context = format_docs(unique_docs)
        input_data = {
            "mbti": self.mbti,
            "feature": self.feature,
            "context": context,
            "query": query,
        }
        chain = self.chat_template | self.llm | StrOutputParser()
        return chain.invoke(input_data)
