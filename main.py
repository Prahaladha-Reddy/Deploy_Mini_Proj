from retrieval import RetrievalSystem
from rag import RAGPipeline

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from chunks import chunks_maker
chunks=chunks_maker(path="output_cleaned.md")
chunks.makedown_splitter()
final_chunks=chunks.recursive_character_splitter()
retrieval_system = RetrievalSystem()
retrieval_system.build_vector_store(final_chunks)  # Your document chunks
rag_pipeline = RAGPipeline(retrieval_system)

# FastAPI app
app = FastAPI(
    title="RAG API",
    description="API for Retrieval-Augmented Generation",
    version="1.0.0"
)

# Request/Response models
class QuestionRequest(BaseModel):
    question: str

class DocumentResponse(BaseModel):
    content: str
    metadata: dict

class AnswerResponse(BaseModel):
    question: str
    answer: str
    relevant_documents: List[DocumentResponse]

# Endpoints
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        # Step 1: Retrieve relevant documents
        retrieved_docs = retrieval_system.retrieve(request.question)
        
        # Step 2: Generate answer
        answer = rag_pipeline.generate_answer(request.question)
        
        # Format documents for response
        documents = [
            DocumentResponse(
                content=doc.page_content,
                metadata=doc.metadata
            )
            for doc in retrieved_docs
        ]
        
        return AnswerResponse(
            question=request.question,
            answer=answer,
            relevant_documents=documents
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# To run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)