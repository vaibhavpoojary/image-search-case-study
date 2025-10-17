import faiss, json, os

def load_faiss_index(index_path="data/faiss.index"):
    index = faiss.read_index(index_path)
    print(f"✅ Loaded FAISS index with {index.ntotal} vectors")
    return index

def load_metadata(meta_path="data/metadata.json"):
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"✅ Loaded metadata for {len(metadata)} images")
    return metadata
