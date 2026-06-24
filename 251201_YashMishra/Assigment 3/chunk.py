from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(text, chunk_size=50, chunk_overlap=10):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)