
import os
import shutil
import uuid
import asyncio
import weaviate
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from typing import Optional
from embed_utils import load_and_split_pdf, get_embeddings
from weaviate_utils import get_client, upsert_docs
from datetime import datetime, timezone
import tempfile
from pydantic import BaseModel
import json
import logging
import traceback
from fastapi import HTTPException


# LLM imports (your original)
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Environment
WEAVIATE_CLASS = os.getenv("WEAVIATE_CLASS", "newlearn")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEAVIATEE_CLASS="newlearn"
client = get_client()
collection = client.collections.get(WEAVIATE_CLASS)

app = FastAPI(title="Weaviate-backed Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


llm = ChatGroq(model="gemma2-9b-it", temperature=0, api_key=GROQ_API_KEY) # ensure env var GROQ_API_KEY set

def weaviate_retrieve(
    question_embedding,
    top_k=3,
    class_name=WEAVIATE_CLASS,
    project_id=None,
    tags=None
):
    """
    Retrieve top_k documents from Weaviate using vector search with optional filters.
    Updated for Weaviate v4 (collections API).
    """
    collection = client.collections.get(class_name)

    # Build filters
    filters = []
    if project_id:
        filters.append({
            "path": ["project_id"],
            "operator": "Equal",
            "valueText": project_id
        })
    if tags:
        filters.append({
            "path": ["tags"],
            "operator": "ContainsAny",
            "valueArray": tags
        })

    where_filter = {"operator": "And", "operands": filters} if filters else None

    # Run query
    res = collection.query.near_vector(
        near_vector=question_embedding,
        limit=top_k,
        filters=where_filter,
        return_properties=["content", "source_uri", "chunk_index", "page_number", "tags"]
    )

    # Extract results safely
    objs = []
    for o in res.objects:
        props = o.properties
        objs.append({
            "content": props.get("content", ""),
            "source_uri": props.get("source_uri", ""),
            "chunk_index": props.get("chunk_index", ""),
            "page_number": props.get("page_number", ""),
            "tags": props.get("tags", []),
        })

    return objs




# Upload endpoint (auto-update + embedding + upsert)
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), project_id: str = Form("proj-123"), tags: str = Form("pdf")):
    """
    Upload a PDF file. The server will:
    - save to temp
    - split into chunks
    - compute embeddings
    - upsert into Weaviate (class newlearn)
    """
    tmpdir = tempfile.mkdtemp()
    file_path = os.path.join(tmpdir, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # load and split
    docs = load_and_split_pdf(file_path)

    # compute embeddings and attach to doc metadata
    for i, d in enumerate(docs):
        vector = get_embeddings(d.page_content)
        # attach vector into metadata so upsert util can pick it
        d.metadata["vector"] = vector
        d.metadata["source"] = file.filename
        d.metadata["document_id"] = f"{uuid.uuid4()}"
        d.metadata["project_id"] = project_id
        d.metadata["tags"] = tags.split(",") if tags else []

    # upsert
    try:
        upsert_docs(WEAVIATEE_CLASS, docs, file, project_id, tags)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # cleanup
        shutil.rmtree(tmpdir, ignore_errors=True)

    return {"status": "success", "uploaded_file": file.filename, "chunks": len(docs)}

# Re-index endpoint (optional)
@app.post("/reindex")
async def reindex(project_id: str = Form(None)):
    """
    Optional: if you want to trigger reindexing from a source directory
    implement your logic here to re-upload existing PDFs or recreate vectors.
    """
    return {"status": "not_implemented", "detail": "Custom reindex logic required."}

# Simple chat endpoint (non-streaming) returns final answer
class ChatRequest(BaseModel):
    question: str
    top_k: int = 3
    project_id: Optional[str] = None
    tags: Optional[str] = None  # comma-separated tags

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        # 1️⃣ Embed the question
        question_embedding = get_embeddings(req.question)

        # 2️⃣ Parse tags into a list
        tags_list = req.tags.split(",") if req.tags else None

        # 3️⃣ Retrieve relevant documents from Weaviate
        objs = weaviate_retrieve(
            question_embedding=question_embedding,
            top_k=req.top_k,
            project_id=req.project_id,
            tags=tags_list
        )

        # 4️⃣ Build context string
        context_parts = [
            f"Source: {o.get('source_uri', 'unknown')} | Page: {o.get('page_number', 'unknown')}\n\n{o.get('content', '')}"
            for o in objs
        ]
        context = "\n\n---\n\n".join(context_parts) if context_parts else ""

        # 5️⃣ Create prompt for LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a knowledgeable and polite support assistant. "
                       "Answer using ONLY the provided context. "
                       "If the answer is not in the context, say: \"I don't have that information.\""),
            ("user", "Context:\n{context}\n\nQuestion: {question}\nAnswer:")
        ])
        chain = LLMChain(llm=llm, prompt=prompt)

        # 6️⃣ Generate LLM response
        answer = chain.run({"context": context, "question": req.question})

        return {"answer": answer, "retrieved_docs": len(objs)}

    except AttributeError as e:
        # Special handling for Weaviate client mismatch
        if "'WeaviateClient' object has no attribute 'query'" in str(e):
            raise HTTPException(
                status_code=500,
                detail="Weaviate client version mismatch: v4 no longer supports client.query. "
                       "Please update weaviate_retrieve() to use collections.query instead."
            )
        raise HTTPException(status_code=500, detail=f"Error in /chat: {str(e)}")

    except Exception as e:
        logging.error("Error in /chat endpoint:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error in /chat: {str(e)}")
