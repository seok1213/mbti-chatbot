from chatbot import MBTIChatBot

# MBTI 목록
mbti_types = ["infp", "enfp", "intj", "entj", "isfj", "esfj", "istp", "estp", "intp", "entp", "isfp", "esfp", "istj", "estj", "infj", "enfj"]

# MBTI 선택 및 챗봇 초기화
selected_mbti = "infp"  # 사용자 입력으로 변경 가능
chatbot = MBTIChatBot(selected_mbti)

# 데이터베이스 초기화 (최초 1회만 필요)
chatbot.initialize_db()
'''
query = '너만의 독특한 습관 있어?'
response = chatbot.get_response(query)
print(response)
'''

# 대화 시작

while True:
    query = input(f"({selected_mbti.upper()}) 질문: ")
    if query.lower() in ["exit", "quit"]:
        print("대화를 종료합니다.")
        break
    response = chatbot.get_response(query)
    print(f"({selected_mbti.upper()}) 답변: {response}")
    