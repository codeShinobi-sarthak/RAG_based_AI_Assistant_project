import os
import joblib
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from google import genai

df = joblib.load("embeddings_df.joblib")


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

    return embeddings


load_dotenv()  # Load environment variables from .env file

def inference(prompt):
    #  setting google api key and calling the google gemini api to get the inference for the above prompt
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )

    return response.text


input_query = input("Enter a query: ")
query_embedding = create_embedding([input_query])[0]
# print(query_embedding)

similarities = cosine_similarity(
    np.vstack(df["embedding"]), [query_embedding] # type: ignore
).flatten()  
# print(similarities)

# to get top results in mathing
top_results = 10
max_idx = similarities.argsort()[::-1][0:top_results]

new_df = df.loc[max_idx]
# for index, item in new_df.iterrows():
#     print(f"chunk_id: {item['chunk_id']}, text: {item['text']}, similarity: {similarities[index]}")


prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{input_query}"

User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''

with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)


#  getting responase from google gemini api for the above prompt and writing it to a text file
get_inference = inference(prompt)
with open("inference.txt", "w", encoding="utf-8") as f:
    f.write(get_inference or "")
