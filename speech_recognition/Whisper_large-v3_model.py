from faster_whisper import WhisperModel

model = WhisperModel("large-v3")

segments, info = model.transcribe("/Users/leonachen/Downloads/intesa_audio.mp4")
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
