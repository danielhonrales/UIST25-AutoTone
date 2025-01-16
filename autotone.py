#import SpeechToTextTranscription.remoteStreamTranscribe as transcriber
#import ContextExtraction.openaiAnalyzer as analyzer
#import VoiceModulation.elevenLabsModulation as modulator
from getApiKey import get_key
from collections import deque
import sounddevice as sd
from openai import OpenAI
import numpy as np
import threading
import asyncio
import wave
import os

# Settings
AUDIO_ID = 777
SAMPLE_RATE = 48000
BLOCK_DURATION = 1
BLOCK_SIZE = SAMPLE_RATE * BLOCK_DURATION
MAX_BLOCKS = 5
CHANNELS = 1
OUTPUT_DIR = os.path.join("participants", f"p{AUDIO_ID}")
BLOCK_ID = 0
BLOCK_COUNTER = 0
SILENCE_THRESHOLD = 0.05
VOICE_ID = -1
BUFFER_SIZE = SAMPLE_RATE * CHANNELS * MAX_BLOCKS

# Setup
openai = OpenAI(
    api_key=get_key()
)
os.makedirs("participants", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
audio_buffer = deque(maxlen=BUFFER_SIZE)
loop = asyncio.get_event_loop()

async def main():
    stream_thread = threading.Thread(target=audio_stream, daemon=True)
    stream_thread.start()
    
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("AutoTone shutting down.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
def audio_stream():
    with sd.InputStream(
        samplerate=SAMPLE_RATE, 
        channels=CHANNELS, 
        blocksize=BLOCK_SIZE,
        dtype="int16",
        callback=audio_callback, 
        ):
        print("Listening... Press Ctrl+C to stop.")
        
        loop.run_forever()

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Error: {status}")
    
    audio_data = indata.flatten()
    audio_buffer.extend(audio_data)
    
    # Process audio
    if VOICE_ID != -1:
        print(f"Modulating voice with voice {VOICE_ID}")
    if detect_silence(indata) or len(audio_buffer) >= BUFFER_SIZE:
        with threading.Lock():
            recent_audio_clip = np.array(audio_buffer)
            audio_buffer.clear()
        asyncio.run_coroutine_threadsafe(process_audio(recent_audio_clip), loop)

def save_audio(block, filename):
    with wave.open(os.path.join(OUTPUT_DIR, filename), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(block)
    #print("Audio block saved")

def detect_silence(audio_data):
    rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
    return rms < SILENCE_THRESHOLD

async def process_audio(audio_data):
    global BLOCK_COUNTER
    BLOCK_COUNTER += 1
    await transcribe_audio(audio_data, BLOCK_COUNTER)
    

def transcribe_audio(audio_data, id):
    print("Transcribing...")
    save_audio(audio_data, f"audio_{id}.wav")
    
    with open(os.path.join(OUTPUT_DIR, f"audio_{id}.wav"), 'rb') as audio_file:
        transcription = openai.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
        print(f"Transcription: {transcription.text}")
        
        with open(os.path.join(OUTPUT_DIR, f"transcript_{id}"), "w+") as transcript_file:
            transcript_file.write(transcription.text)

asyncio.run(main())