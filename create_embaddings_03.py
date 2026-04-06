import json
import joblib
import requests
import pandas as pd

def create_embedding(text_list):
    """Call the Ollama embed API and return just the embeddings list."""
    resp = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "bge-m3", "input": text_list},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()

    # API returns { "embeddings": [[...], [...]] }
    embeddings = payload.get("embeddings")
    if embeddings is None:
        raise ValueError(f"Unexpected response from embed API: {payload}")
    if len(embeddings) != len(text_list):
        raise ValueError(
            f"Mismatch between texts ({len(text_list)}) and embeddings ({len(embeddings)})"
        )
    return embeddings



with open("outputs/05_Image, Lists, and Tables in HTML.mp3.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

segments = chunks["segments"]
segment_texts = [segment["text"].strip() for segment in segments] # len(segment_texts) = 829

embeddings = []
embeddings.extend(create_embedding(segment_texts))

for idx, (segment, embedding) in enumerate(zip(segments, embeddings)):
    segment["chunk_id"] = idx
    segment["embedding"] = embedding

df = pd.DataFrame(segments)
joblib.dump(df, "embeddings_df.joblib")
print(df.head())
print("df saved by joblib.dump()")


# Find similarities of question_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)
