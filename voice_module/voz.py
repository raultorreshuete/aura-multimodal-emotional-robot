import sounddevice as sd
import numpy as np
import whisper
import pyttsx3
import time
import json
import requests
from http import HTTPStatus

from collections import deque

#pip install git+https://github.com/openai/whisper.git sounddevice numpy

# Configuración
samplerate = 16000
chunk_size = 1024  # muestras por bloque (~64 ms a 16kHz)
silence_duration = 2  # segundos de silencio para finalizar grabación
silence_threshold = 0.01  # volumen RMS para considerar silencio

model = whisper.load_model("turbo")
tts = pyttsx3.init()
tts.setProperty('rate', 150)

def rms(audio_chunk):
    return np.sqrt(np.mean(audio_chunk**2))

def responder(texto):
    print("🗣️ Respondiendo:", texto)
    tts.say(texto)
    tts.runAndWait()

def escuchar_hasta_silencio():
    buffer = []
    silencio_contador = 0
    print("🎧 Escuchando... (habla cuando quieras)")
    
    with sd.InputStream(samplerate=samplerate, channels=1, blocksize=chunk_size, dtype='float32') as stream:
        while True:
            chunk, _ = stream.read(chunk_size)
            chunk = np.squeeze(chunk)
            buffer.append(chunk)

            # Comprobar si es silencio
            energia = rms(chunk)
            if energia < silence_threshold:
                silencio_contador += chunk_size / samplerate
            else:
                silencio_contador = 0

            # Si hay 2 segundos de silencio seguidos, terminamos
            if silencio_contador >= silence_duration:
                break

    audio_total = np.concatenate(buffer)
    return audio_total

def transcribir_audio(audio):
    print("🧠 Transcribiendo...")
    result = model.transcribe(audio, language="es")
    texto = result['text'].strip()
    print("📝 Tú dijiste:", texto)
    response = process_audio (texto)
    return response

def process_audio (msg: str) -> str:
        
        """
        Function that processes the auido transcripted.
        
        Parameters:
            - text (str): audio transcripted that has to be processed
        
        Returns: 
            - str: the response to the audio processed
        """
        url = "http://localhost:50007/recepcion-voz"
        sender = "test_user_voice"
        
        headers = {
            'Content-Type': 'application/json'
        }

        payload = {
            "sender" : sender,
            "message" : msg
        }
        payload = json.dumps (payload)
        response = requests.post (url= url, data = payload, headers=headers)
        if (response.status_code == HTTPStatus.OK ):
            response_json = response.text  
            return response_json
        else:
            return "There has been an error."

def main():
    print("🤖 Asistente de voz iniciado.")

    while True:
        audio = escuchar_hasta_silencio()
        if rms(audio) < silence_threshold:
            continue  # ignorar si todo fue silencio

        texto = transcribir_audio(audio).lower()
        if "adiós" in texto:
            responder("Hasta luego.")
            print("👋 Programa finalizado.")
            break

        responder(texto)

if __name__ == "__main__":
    main()