import torch, numpy as np

def search_images(model, device, index, metadata, query, top_k=5):
    with torch.no_grad():
        tokens = model.tokenize([query]).to(device)
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        q = text_features.cpu().numpy().astype("float32")

    scores, ids = index.search(q, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        results.append({
            "filename": metadata[idx]["filename"],
            "path": metadata[idx]["path"],
            "score": float(score)
        })
    return results
