import json
import whisper

model = whisper.load_model("large-v2")
audio = whisper.load_audio("audios/Matrices.mp3")
result = model.transcribe(audio, language="hi", task="translate")

print(result["segments"])

chunks = []
for segment in result["segments"]:
    chunks.append(
        {"start": segment["start"], "end": segment["end"], "text": segment["text"]} # type: ignore
    )

print(chunks)

with open("whisper_output.json", "w") as f:
    json.dump(chunks, f, indent=2)


# audios = os.listdir('audios')

# for audio in audios:
#     audio = whisper.load_audio(os.path.join('audios', audio))
#     result = model.transcribe(audio, language="hi", translate=True)
#     print(result["text"])
# print("Done")
