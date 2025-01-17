import os
import speech_recognition as sr
import pyttsx3
from chat import initialize_query_engine
import pyaudio

def recognize_audio():
    """Converts audio input into text using SpeechRecognition."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            print("Processing audio...")
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError as e:
            return f"Error with STT service: {e}"

def speak_text(text):
    """Converts text into audio using pyttsx3."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def voicebot(user_id:str):
    query_engine = initialize_query_engine(user_id)
    while True:
        print("Say something or 'exit' to quit:")
        user_query = recognize_audio()
        if user_query.lower() == "exit":
            print("Exiting voice bot.")
            speak_text("Goodbye!")
            break
        
        print(f"Query: {user_query}")
        response = query_engine.query(user_query)
        print(f"Response: {response.response}")
        speak_text(response.response)

if __name__ == "__main__":
    voicebot("admin")