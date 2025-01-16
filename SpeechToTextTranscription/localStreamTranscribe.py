import sounddevice as sd
import numpy as np
import whisper

# Load Whisper model
model = whisper.load_model("base")

# Audio settings
SAMPLE_RATE = 16000
BLOCK_SIZE = int(SAMPLE_RATE / 4)  # Process audio in 250ms chunks

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Error: {status}")
    # Normalize audio
    audio = np.squeeze(indata).astype(np.float32)
    # Perform transcription
    result = model.transcribe(audio, fp16=False)
    print(f"Transcription: {result['text']}")

# Start audio stream
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=BLOCK_SIZE):
    print("Streaming... Press Ctrl+C to stop.")
    sd.sleep(30 * 1000)  # Stream for 1 minute