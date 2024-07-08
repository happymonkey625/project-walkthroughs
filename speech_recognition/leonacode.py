import os
from pydub import AudioSegment
import speech_recognition as sr
from transformers import pipeline

audio_file_path = "/Users/leonachen/Downloads/intesa_audio.mp4"


def convert_to_wav(input_file):
    output_file = "converted_audio.wav"
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format="wav")
    return output_file

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        transcription = recognizer.recognize_google(audio)
        return transcription
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # Adjust max_length and min_length dynamically
    input_length = len(text.split())
    max_length = min(150, input_length - 5) if input_length > 10 else input_length
    min_length = min(30, max_length - 10) if max_length > 20 else max_length
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    audio_file_path = "/Users/leonachen/Downloads/intesa_audio.mp4"  # Update with your actual file path
    converted_audio_file_path = convert_to_wav(audio_file_path)
    
    transcription = transcribe_audio(converted_audio_file_path)
    os.remove(converted_audio_file_path)  # Clean up the converted file

    if "Google Speech Recognition could not understand audio" not in transcription:
        summary = summarize_text(transcription)
        print("Transcription:\n", transcription)
        print("\nSummary:\n", summary)
    else:
        print(transcription)