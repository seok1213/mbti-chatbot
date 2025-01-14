from flask import Flask, render_template, request, jsonify  # Flask 관련 모듈 불러오기 (웹 애플리케이션 생성 및 요청/응답 처리)
import openai  # OpenAI API 호출을 위한 모듈
import json  # JSON 데이터 처리 모듈
import os  # OS 관련 기능 (환경 변수 처리)
from dotenv import load_dotenv  # .env 파일에서 환경 변수 로드

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

# 채팅 API (RAG 방식)
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

    # OpenAI API 호출
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 사용할 OpenAI 모델
            messages=[
                {"role": "system", "content": f"당신은 {user_mbti} 성격을 가진 AI입니다."},  # 시스템 메시지 (AI 역할 설정)
                {"role": "user", "content": f"사용자가 입력한 메시지: '{user_message}'\n관련 데이터: '{retrieved_data}'"}  # 사용자 메시지와 검색된 데이터 전달
            ],
            max_tokens=300,  # 생성할 응답의 최대 토큰 수
            temperature=0.7  # 응답의 다양성 조정
        )
        reply = response['choices'][0]['message']['content'].strip()  # AI 응답 내용 추출
    except Exception as e:
        reply = f"AI 응답 생성 중 문제가 발생했습니다: {str(e)}"  # 예외 발생 시 오류 메시지 반환

    # 대화 기록 업데이트
    conversation_history.append({"user": user_message, "assistant": reply})  # 대화 내용 저장
    return jsonify({"reply": reply})  # AI 응답 반환

# Flask 실행
if __name__ == '__main__':
    app.run(debug=True)  # 디버그 모드에서 Flask 앱 실행