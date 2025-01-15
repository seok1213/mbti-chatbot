from flask import Flask, render_template, request, jsonify
import openai
from dotenv import load_dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Flask 앱 초기화
app = Flask(__name__)

# OpenAI Embeddings 초기화
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 사용자 MBTI와 대화 기록을 저장할 전역 변수
user_mbti = None  # 현재 사용자의 MBTI
conversation_history = []  # 대화 기록 저장 리스트

# 기본 페이지 라우트
@app.route('/')
def home():
    return render_template('index.html')

# MBTI 설정 API
@app.route('/set_mbti', methods=['POST'])
def set_mbti():
    global user_mbti, conversation_history
    data = request.json
    user_mbti = data.get('mbti', '').strip().upper()  # 사용자가 설정한 MBTI
    conversation_history = []  # 대화 기록 초기화
    return jsonify({"reply": f"MBTI가 {user_mbti}로 설정되었습니다."})

# 채팅 API
@app.route('/chat', methods=['POST'])
def chat():
    global user_mbti, conversation_history
    user_message = request.json.get('message', '').strip()

    if not user_mbti:
        return jsonify({"reply": "먼저 MBTI를 선택해주세요."})

    if not user_message:
        return jsonify({"reply": "메시지를 입력해주세요!"})

    # ChromaDB에서 사용자 MBTI 컬렉션 접근
    persist_directory = f"./chromadb/{user_mbti}"
    try:
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    except Exception:
        return jsonify({"reply": f"'{user_mbti}'에 대한 데이터를 찾을 수 없습니다."})

    results = vectorstore.similarity_search(query=user_message, k=1)
    if not results:
        # 검색된 정보가 없을 때 GPT가 역할에 맞게 답변 생성
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"너는 {user_mbti} 성격 유형을 가진 친구야. 질문에 간단하고 직관적이게 반말로 짧게 말해."},
                    {"role": "user", "content": f"질문: {user_message}\n하지만 관련된 정보는 없어. 질문에 대해서 적절하게 성격에 맞게 짧고 간단하게 답변해줘."}      
                ],
                max_tokens=200,
                temperature=0.1
            )
            reply = response['choices'][0]['message']['content'].strip()
        except Exception as e:
            reply = f"AI 응답 생성 중 문제가 발생했습니다: {str(e)}"
    else:
        # 검색 결과가 있을 때 GPT에 검색된 정보를 제공
        retrieved_text = results[0].page_content[:100]
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"너는 {user_mbti} 성격 유형을 가진 친구야. 질문에 대해 간단하고 직관적으로, 반말로 짧게 대답해줘. 검색된 정보가 있으면 자연스럽게 포함해. 너가 {user_mbti} 실제로 성격을 가진 것처럼 답해야해 말투도 성격에 맞게 소심하면 소심하게 외향적인 성격이면 당차게 바꿔서 답해"},
                    {"role": "user", "content": f"질문: {user_message}\n검색된 정보: {retrieved_text}"}
                ],
                max_tokens=200,
                temperature=0.1
            )
            reply = response['choices'][0]['message']['content'].strip()
        except Exception as e:
            reply = f"AI 응답 생성 중 문제가 발생했습니다: {str(e)}"

    conversation_history.append({"user": user_message, "assistant": reply})
    return jsonify({"reply": reply, "history": conversation_history})

# Flask 서버 실행
if __name__ == '__main__':
    app.run(debug=True)
