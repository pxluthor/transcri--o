import whisper

modelo = whisper.load_model("tiny")

resposta = modelo.transcribe("audio_12m.mp3")

print(resposta["text"])