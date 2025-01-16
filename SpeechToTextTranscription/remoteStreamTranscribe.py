import sounddevice as sd
import numpy as np
import wave
import time as timePack
import os
from openai import OpenAI
from getApiKey import get_key

# Settings
AUDIO_ID = 777
SAMPLE_RATE = 8000
BLOCK_DURATION = 5
BLOCK_SIZE = SAMPLE_RATE * BLOCK_DURATION
CHANNELS = 1
BLOCK_OUTPUT_DIR = os.path.join("audio_blocks", f"P{AUDIO_ID}")
TRANSCRIPT_OUT_DIR = os.path.join("audio_transcripts", f"P{AUDIO_ID}")
BLOCK_ID = 0

# Setup
openai = OpenAI(
    api_key=get_key()
)
os.makedirs("audio_blocks", exist_ok=True)
os.makedirs("audio_transcripts", exist_ok=True)
os.makedirs(BLOCK_OUTPUT_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_OUT_DIR, exist_ok=True)
transcript_file = open(os.path.join(TRANSCRIPT_OUT_DIR, "transcript"), "w+")

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Error: {status}")
    
    audio_data = indata.tobytes()
    
    # Save audio chunk
    block_filename = f"block-{timePack.strftime('%H_%M_%S')}.wav"
    save_audio_block(audio_data, block_filename)
    
    # Process audio with OpenAI
    process_audio(block_filename)

# Function to save audio chunks
def save_audio_block(chunk, filename):
    with wave.open(os.path.join(BLOCK_OUTPUT_DIR, filename), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(chunk)
    #print("Audio block saved")

def process_audio(file_path):
    with open(os.path.join(BLOCK_OUTPUT_DIR, file_path), 'rb') as audio_file:
        transcription = openai.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
        print(transcription.text)
        transcript_file.write(transcription.text)

########################################################################################################################################

# Start audio stream
try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE, 
        channels=CHANNELS, 
        blocksize=BLOCK_SIZE,
        dtype="int16",
        callback=audio_callback, 
        ):
        print("Streaming... Press Ctrl+C to stop.")
        
        sd.sleep(BLOCK_DURATION * 10 * 1000)
except KeyboardInterrupt:
    print("Streaming stopped")