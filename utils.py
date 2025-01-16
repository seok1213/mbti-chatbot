from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

def format_docs(docs):
    # 도큐먼트 포맷팅
    return '\n\n'.join([d.page_content for d in docs])
