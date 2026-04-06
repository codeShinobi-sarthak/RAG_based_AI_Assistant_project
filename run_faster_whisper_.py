import json
# import os
import time

# We use faster_whisper because it runs on CTranslate2 (C++ backend)
# This is much faster on laptops than standard PyTorch-based whisper.
from faster_whisper import WhisperModel

# --- CONFIGURATION ---
# "large-v3" is the most accurate model.
# "distil-large-v3" is a good alternative if this is still too slow.
MODEL_SIZE = "medium"

print("--> Loading Model...")
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")


def process_audio(file_path):
    """
    Transcribes a single audio file and returns the chunks.
    """
    print(f"--> Starting transcription for: {file_path}")
    # checking how much time it take
    start_time = time.time()

    # .transcribe() in faster-whisper is different from standard whisper.
    # It returns a 'generator' (segments) instead of a full list.
    # This means it doesn't process the whole file instantly; it waits for you to loop.
    # beam_size=5: Looks at 5 prominent possibilities for each sentence to reduce errors.
    segments, info = model.transcribe(
        file_path,
        language="hi",  # Force Hindi (improves accuracy for Hindi audio)
        task="translate",  # Translates Hindi Audio -> English Text directly
        beam_size=5,
    )

    print(
        f"    Detected language '{info.language}' with probability {info.language_probability:.2f}"
    )

    chunks = []

    # The actual processing happens HERE in this loop.
    # Because 'segments' is a generator, we process one sentence at a time.
    # This prevents your RAM from overflowing on 1-hour videos.
    for segment in segments:
        chunk = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
        }
        chunks.append(chunk)
        # Optional: Print to see progress so you know it's not stuck
        # print(f"    {segment.text[:50]}...")

    duration = time.time() - start_time
    print(f"--> Finished {file_path} in {duration:.2f} seconds.")
    return chunks


#  doing it for only one file to test it out. We will loop through all files later.
file_chunks = process_audio("audios/Matrices.mp3")
with open("faster_whisper_output.json", "w", encoding="utf-8") as f:
    json.dump(file_chunks, f, indent=2, ensure_ascii=False)


"""
# --- MAIN EXECUTION ---
audio_folder = "audios"
output_file = "all_transcriptions.json"
all_data = []

if os.path.exists(audio_folder):
    # Filter for common audio/video formats
    files = [
        f
        for f in os.listdir(audio_folder)
        if f.endswith((".mp3", ".mp4", ".wav", ".m4a"))
    ]
    print(f"Found {len(files)} files to process.")

    for filename in files:
        full_path = os.path.join(audio_folder, filename)

        # We collect all chunks for this specific file
        file_chunks = process_audio(full_path)

        # We structure the data for RAG.
        # 'filename' is needed so the AI knows WHICH video the text came from.
        all_data.append({"source": filename, "chunks": file_chunks})

    # Save everything to one JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"\nAll done! Saved to {output_file}")
else:
    print(f"Error: Folder '{audio_folder}' not found.")
"""
