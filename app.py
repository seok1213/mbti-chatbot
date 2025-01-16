import streamlit as st
from chatbot import MBTIChatBot

# MBTI 목록
mbti_types = [
    "infp", "enfp", "intj", "entj",
    "isfj", "esfj", "istp", "estp",
    "intp", "entp", "isfp", "esfp",
    "istj", "estj", "infj", "enfj"
]

st.title("MBTI Chatbot")

# 1) 사이드바에서 MBTI 유형 선택
selected_mbti = st.sidebar.selectbox("Select MBTI type", mbti_types, index=0)

# 2) 세션 스테이트에 챗봇이 없으면 초기화
if "chatbot" not in st.session_state:
    st.session_state["chatbot"] = MBTIChatBot(selected_mbti)
    st.session_state["chatbot"].initialize_db()
    # 초기 대화 메시지도 함께 설정
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": f"안녕! 내 mbti는 {selected_mbti.upper()} 야!"
        }
    ]

# 3) 만약 MBTI가 변경되었다면, 기존 대화/챗봇을 초기화
if st.session_state["chatbot"].mbti != selected_mbti:
    # 새 챗봇 생성
    st.session_state["chatbot"] = MBTIChatBot(selected_mbti)
    st.session_state["chatbot"].initialize_db()

    # 대화 내용을 모두 리셋(새로운 MBTI 시작)
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": f"안녕! 내 mbti는 {selected_mbti.upper()} 야!"
        }
    ]

# 4) 지금까지의 대화(메시지) 표시
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 6) 사용자 입력
if user_input := st.chat_input("질문을 입력해주세요 :)"):
    # 화면에 사용자 메시지 출력
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # [스트리밍 모드]로 챗봇의 응답 가져오기
    response = st.session_state["chatbot"].get_response(user_input)

    # 한 글자씩 받아서 출력할 자리(placeholder)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        streamed_text = ""

        # spinner로 표시할 수도 있음
        with st.spinner("생각 중..."):
            for chunk in response:
                streamed_text += chunk
                # 매 chunk마다 placeholder를 업데이트하여 타이핑 효과
                message_placeholder.write(streamed_text)

    # 스트리밍이 끝난 최종 응답을 messages에 저장
    st.session_state["messages"].append(
        {"role": "assistant", "content": streamed_text}
    )