from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from chunk import create_chunks

app = FastAPI(
    title="Text Chunking API",
    description="FastAPI Assignment 3 - Text Chunking using LangChain",
    version="1.0.0"
)


class TextRequest(BaseModel):
    text: str
    chunk_size: int = 50
    chunk_overlap: int = 10


@app.get("/")
def welcome():
    return {
        "message": "Welcome to Text Chunking API"
    }


@app.post("/chunk")
def chunk_text(request: TextRequest):

    # Validate text input
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    # Validate chunk size
    if request.chunk_size <= 0:
        raise HTTPException(
            status_code=400,
            detail="chunk_size must be positive"
        )

    # Validate chunk overlap
    if request.chunk_overlap < 0:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap cannot be negative"
        )

    if request.chunk_overlap >= request.chunk_size:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap must be less than chunk_size"
        )

    try:
        chunks = create_chunks(
            request.text,
            request.chunk_size,
            request.chunk_overlap
        )

        return {
            "chunks": chunks,
            "total_chunks": len(chunks)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while chunking text: {str(e)}"
        )