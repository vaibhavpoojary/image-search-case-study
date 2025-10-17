import faiss
import numpy as np

def build_faiss_index(
    embeddings_path="data/clip_image_embeddings.npy",
    output_path="data/faiss.index"
):
    # Load embeddings
    embeddings = np.load(embeddings_path).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create FAISS index (FlatIP for cosine search)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save the index
    faiss.write_index(index, output_path)
    print(f"✅ FAISS index built and saved at {output_path}")
    print(f"Index contains {index.ntotal} vectors of dimension {dim}")

if __name__ == "__main__":
    build_faiss_index()
