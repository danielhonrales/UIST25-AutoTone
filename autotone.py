#import SpeechToTextTranscription.remoteStreamTranscribe as transcriber
#import ContextExtraction.openaiAnalyzer as analyzer
#import VoiceModulation.elevenLabsModulation as modulator
from getApiKey import get_key
import scipy.io.wavfile
import sounddevice as sd
from openai import OpenAI
import numpy as np
import wave
import os

# Settings
AUDIO_ID = 777
SAMPLE_RATE = 44100
BLOCK_DURATION = 1
BLOCK_SIZE = SAMPLE_RATE * BLOCK_DURATION
MAX_BLOCKS = 15
CHANNELS = 1
BLOCK_OUTPUT_DIR = os.path.join("participants", f"p{AUDIO_ID}")
BLOCK_ID = 0
BLOCK_COUNTER = 0
SILENCE_THRESHOLD = 0.05

# Setup
openai = OpenAI(
    api_key=get_key()
)
os.makedirs("participants", exist_ok=True)
os.makedirs(BLOCK_OUTPUT_DIR, exist_ok=True)
audio_blocks = []

def main():
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, 
            channels=CHANNELS, 
            blocksize=BLOCK_SIZE,
            dtype="int16",
            callback=audio_callback, 
            ):
            print("Streaming... Press Ctrl+C to stop.")
            
            while True:
                sd.sleep(1)
                
    except KeyboardInterrupt:
        print("Streaming stopped")
    except Exception as e:
        print(f"An error occurred: {e}")

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Error: {status}")
    
    audio_data = indata.tobytes()
    
    # Save audio block
    global BLOCK_COUNTER
    block_filename = f"block_{BLOCK_COUNTER}.wav"
    save_audio_block(audio_data, block_filename)
    BLOCK_COUNTER += 1
    audio_blocks += block_filename
    
    # Process audio
    ## If voice selected, modulate
    if detect_silence(audio_data) or len(audio_blocks) >= 15:
        combined_blocks_file = concatenate_wav(audio_blocks)
        audio_blocks = []

def save_audio_block(block, filename):
    with wave.open(os.path.join(BLOCK_OUTPUT_DIR, filename), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(block)
    #print("Audio block saved")

def detect_silence(audio_data):
    rms = np.sqrt(np.mean(np.square(audio_data)))
    return rms < SILENCE_THRESHOLD

def concatenate_wav(wav_list):
    def extract_integers(input_string):
        return ''.join(char for char in input_string if char.isdigit())
    
    combined_data = []
    for wav_file in wav_list:
        sr, data = scipy.io.wavfile.read(os.path.join(BLOCK_OUTPUT_DIR, wav_file))
        
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.max(np.abs(data))
        combined_data.append(data)
    
    combined_data = np.concatenate(combined_data)
    combined_data = (combined_data * 32767).astype(np.int16)
    combined_data_file_name = f"blocks_{extract_integers(wav_list[0])}-{extract_integers(wav_list[-1])}"
    with wave.open(os.path.join(BLOCK_OUTPUT_DIR, combined_data_file_name), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(combined_data)
    
    return combined_data_file_name

main()