# RAG-Based AI Assistant

A small pipeline that turns course videos into searchable text with embeddings and a simple query script.

## Project Structure
- `videos/` — raw video files.
- `audios/` — audio tracks extracted from videos.
- `jsons/` and `outputs/` — transcriptions and enriched segment JSON files.
- `embeddings_df.joblib` — persisted embeddings DataFrame.
- Key scripts:
  - `02_process_videos.py` — convert videos to MP3 with FFmpeg.
  - `run_faster_whisper_.py` — transcribe a single audio with faster-whisper (CUDA); includes a commented loop template for batches.
  - `01_run_whisper_create_chunks.py` — example using openai-whisper to transcribe one audio and write `whisper_output.json`.
  - `read_chunks_03.py` — load a segment JSON, fix sequential `chunk_id`, call Ollama `bge-m3` to embed each segment, and save `embeddings_df.joblib`.
  - `04_process_query.py` — prompt for a query, embed it, and return the top similar segments by cosine similarity.
  - `fix_chunk_ids.py` — standalone helper to normalize `chunk_id` values in a JSON file.

## Prerequisites
- Python 3.10+
- FFmpeg installed and `ffmpeg_path` in `02_process_videos.py` updated for your system.
- GPU with CUDA for faster-whisper (or change device/compute_type).
- Ollama running locally with the `bge-m3` model pulled (`ollama pull bge-m3`).
- Python packages: `whisper`, `faster-whisper`, `requests`, `pandas`, `joblib`, `numpy`, `scikit-learn`.

## Typical Workflow
1) Convert videos to audio
- Place source videos in `videos/`.
- Run `python 02_process_videos.py` to create MP3s in `audios/` (adjust `ffmpeg_path` first).

2) Transcribe
- For GPU, run `python run_faster_whisper_.py` .
- Or use `01_run_whisper_create_chunks.py` (PyTorch whisper) for a sample single-file transcription.
- Save or move the resulting segment JSON into `outputs/` (ensure it has a `segments` list with `text` fields).

3) Build embeddings
- Update `JSON_PATH` inside `create_embaddings_03.py` to point to your target segment JSON.
- Run `python create_embaddings_03.py` to rewrite sequential `chunk_id`s, embed each segment via Ollama, and save `embeddings_df.joblib`.

4) Query the corpus
- Run `python 04_process_query.py`.
- Enter a question; the script embeds it and prints the top matching segments.
- you can use the top number you want like 3, 5, 10

