from flask import Flask, render_template, request, jsonify  # Flask 관련 모듈 불러오기 (웹 애플리케이션 생성 및 요청/응답 처리)
import openai  # OpenAI API 호출을 위한 모듈
import json  # JSON 데이터 처리 모듈
import os  # OS 관련 기능 (환경 변수 처리)
from dotenv import load_dotenv  # .env 파일에서 환경 변수 로드
from langchain.prompts import PromptTemplate  # LangChain의 프롬프트 템플릿 모듈
from langchain.chains import LLMChain  # LangChain의 체인 모듈
from langchain.chat_models import ChatOpenAI  # OpenAI를 사용하는 LangChain 챗 모델

# .env 파일 로드
load_dotenv("OpenAI_key.env")  # OpenAI API 키가 포함된 .env 파일 로드
openai.api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 OpenAI API 키 가져오기

# Flask 애플리케이션 생성
app = Flask(__name__)

# MBTI 데이터 로드
with open('mbti_data.json', 'r', encoding='utf-8') as file:
    mbti_data = json.load(file)  # MBTI 관련 JSON 데이터를 메모리에 로드

# 사용자 MBTI와 대화 기록을 저장할 전역 변수
user_mbti = None  # 현재 사용자의 MBTI
conversation_history = []  # 대화 기록 저장 리스트

# LangChain을 사용한 LLM 설정
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)  # OpenAI 모델 설정

# 프롬프트 템플릿 설정
prompt_template = PromptTemplate(
    input_variables=["mbti", "user_input", "retrieved_data"],
    template=(
        "당신은 {mbti} 성격 유형을 가진 AI입니다.\n"
        "사용자가 입력한 메시지: '{user_input}'\n"
        "관련 데이터: '{retrieved_data}'\n"
        "위 정보를 바탕으로 친절하고 자세한 답변을 생성하세요."
    )
)

# LangChain 체인 생성
chat_chain = LLMChain(llm=llm, prompt=prompt_template)

# JSON 데이터 검색 함수
def search_mbti_data(mbti, user_input):
    """사용자의 입력과 일치하는 MBTI 데이터를 검색"""
    if mbti in mbti_data:  # 선택한 MBTI가 데이터에 있는지 확인
        for keyword, response in mbti_data[mbti].get('keywords', {}).items():  # 키워드와 응답 매핑
            if keyword in user_input:  # 사용자의 입력에 키워드가 포함된 경우
                return response  # 해당 키워드에 대응하는 응답 반환
    return None  # 일치하는 데이터가 없을 경우 None 반환

# 기본 페이지 라우트
@app.route('/')
def home():
    return render_template('index.html')  # UI 템플릿 렌더링 (기본 화면)

# MBTI 설정 API
@app.route('/set_mbti', methods=['POST'])
def set_mbti():
    global user_mbti, conversation_history  # 전역 변수 접근
    data = request.json  # JSON 형식으로 데이터 받기
    user_mbti = data.get('mbti')  # 사용자가 설정한 MBTI 값 저장
    conversation_history = []  # 대화 기록 초기화
    return jsonify({"reply": f"MBTI가 {user_mbti}로 설정되었습니다."})  # 설정 결과 응답

# 채팅 API
@app.route('/chat', methods=['POST'])
def chat():
    global user_mbti, conversation_history  # 전역 변수 접근

    user_message = request.json.get('message', '').strip()  # 사용자가 보낸 메시지 가져오기

    # MBTI가 설정되지 않은 경우
    if not user_mbti:
        return jsonify({"reply": "먼저 MBTI를 선택해주세요."})

    # 빈 메시지 처리
    if not user_message:
        return jsonify({"reply": "메시지를 입력해주세요!"})

    # JSON 데이터에서 검색
    retrieved_data = search_mbti_data(user_mbti, user_message)

    # LangChain을 통해 OpenAI API 호출
    try:
        reply = chat_chain.run({
            "mbti": user_mbti,
            "user_input": user_message,
            "retrieved_data": retrieved_data or "관련 데이터 없음"
        })
    except Exception as e:
        reply = f"AI 응답 생성 중 문제가 발생했습니다: {str(e)}"

    # 대화 기록 업데이트
    conversation_history.append({"user": user_message, "assistant": reply})  # 대화 내용 저장
    return jsonify({"reply": reply})  # AI 응답 반환

# Flask 실행
if __name__ == '__main__':
    app.run(debug=True)  # 디버그 모드에서 Flask 앱 실행
