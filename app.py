from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI Chat 모델 초기화
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0)

# 텍스트 파일 로더 초기화
loader = TextLoader(r"mbti_texts/INFP.txt", encoding="utf-8")
data = loader.load()

# 텍스트 분할기 초기화
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    length_function=len
)
texts = text_splitter.split_documents(data)

# OpenAI 임베딩 초기화
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

mbti = 'infp'

# 벡터스토어 초기화
persist_directory = f"./chromadb/{mbti}"
db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.8})

# 도큐먼트 포맷팅 함수
def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

# INFP 특징
feature = '''
INFP는 내향적이면서도 상상력이 풍부한 성격으로, 자신만의 가치와 내면 세계를 탐구하는 것을 중요하게 생각해.  
그들은 주로 조용한 환경에서 혼자만의 시간을 가지며, 자신을 이해하고 성장하려는 노력을 많이 해.  
타인의 감정을 깊이 공감하고, 그들에게 도움을 주는 일을 통해 보람을 느껴.  
예술적이고 창의적인 활동을 좋아하며, 글쓰기, 음악, 그림과 같은 표현 활동에서 큰 만족을 얻어.  
세상의 아름다움과 자연 속에서 영감을 받으며, 단순히 존재하는 것만으로도 의미를 느끼는 편이야.  
이상주의적인 경향이 강해서, 세상을 더 나은 곳으로 만들고 싶다는 생각을 자주 해.  
'''

# 대화 히스토리 관리
dialogue_history = []

def reset_dialogue_history():
    global dialogue_history
    dialogue_history = []

# 대화 내용을 히스토리에 추가하는 함수
def add_to_dialogue_history(user_input, ai_response):
    dialogue_history.append(f"사용자: {user_input}")
    dialogue_history.append(f"AI: {ai_response}")

# 대화 루프
while True:
    user_input = input("질문: ")

    # 종료 명령 처리
    if user_input.lower() in ["종료", "quit", "exit"]:
        print("프로그램을 종료합니다.")
        break

    # 초기화 명령 처리
    if user_input.lower() in ["초기화", "reset", "새로운 대화 시작"]:
        reset_dialogue_history()
        print("대화 히스토리가 초기화되었습니다.")
        continue

    # 관련 문서 검색
    relevant_docs = retriever.get_relevant_documents(user_input)
    context = format_docs(relevant_docs)

    # 채팅 템플릿 초기화
    chat_template = ChatPromptTemplate.from_template(
        """
        '''
        대화 히스토리
        {history}
        '''
        '''
        mbti
        {mbti}
        '''
        '''
        특징
        {feature}
        '''
        '''
        대화 예시
        {context}
        '''
        '''
        질문
        {query}
        '''

        너는 {mbti} 성격을 가진 사람처럼 행동해야 해.
        아래 {feature}를 참고해서 {mbti}의 성격을 바탕으로 자연스럽고 일상적인 대화를 이어가줘.
        대답은 반드시 가볍고 편안한 반말로 하고, 짧고 직관적이게 답해. 사용자는 너랑 같은 성격을 가지고 있지 않아.
        """
    )

    # 대화 히스토리 텍스트 생성
    history_text = "\n".join(dialogue_history)

    # 입력 데이터 준비
    input_data = {
        "history": history_text,
        "mbti": mbti,
        "feature": feature,
        "context": context,
        "query": user_input
    }

    # 체인 실행
    chain = (
        chat_template | llm | StrOutputParser()
    )
    ai_response = chain.invoke(input_data)

    # 결과 출력
    print(f"답변 : {ai_response}")

    # 대화 히스토리에 추가
    add_to_dialogue_history(user_input, ai_response)
