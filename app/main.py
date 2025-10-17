from fastapi import FastAPI, Query
from app.models.clip_loader import load_clip_model
from app.services.indexer import load_faiss_index, load_metadata
from app.services.search_engine import search_images

app = FastAPI(title="Image Search API")

# Load once on startup
model, preprocess, device = load_clip_model()
index = load_faiss_index("data/faiss.index")
metadata = load_metadata("data/metadata.json")

@app.get("/api/v1/search")
def search(query: str = Query(..., description="Natural language search query"), top_k: int = 5):
    results = search_images(model, device, index, metadata, query, top_k)
    return {"query": query, "results": results}

@app.get("/api/v1/health")
def health_check():
    return {"status": "ok"}
