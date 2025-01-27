from elevenlabs import ElevenLabs
from getApiKey import get_key
from collections import deque
from openai import OpenAI
import numpy as np
import pyaudio
import asyncio
import wave
import os


# Set up the PyAudio stream parameters
AUDIO_ID = 777
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Sample rate (samples per second)
CHUNK = 1024  # Number of frames per buffer
SILENCE_THRESHOLD = 500
API_URL = "https://your-api.com/process_audio"  # Replace with your API endpoint
OUTPUT_DIR = os.path.join("participants", f"p{AUDIO_ID}")


# Asyncio Queue for communication between the callback and worker
openai = OpenAI(
    api_key=get_key("apiKey_openai")
)
elevenlabs = ElevenLabs(
    api_key=get_key("apiKey_elevenlabs")
)
os.makedirs("participants", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
audio_queue = asyncio.Queue()

# Flag to indicate if processing is required
PROCESS_AUDIO = False

# Simulated function for making the API request and processing audio
async def process_audio_with_api(audio_data):
    try:
        async with asyncio.to_thread(transcribe_audio(audio_data, AUDIO_ID)) as transcription:
            print("Transcription: " + transcription)
    except Exception as e:
        print("Transcription failed, " + str(e))


# Async worker that processes audio requests from the queue
async def api_worker():
    while True:
        # Get audio data from the queue (this will block if the queue is empty)
        audio_data = await audio_queue.get()

        if audio_data is None:
            break  # Exit thread if None is received

        # Process the audio with the API
        processed_audio = await process_audio_with_api(audio_data)
        
        # Put the processed audio back into the queue for the callback
        await audio_queue.put(processed_audio)

# Callback function for processing and streaming audio
def callback(in_data, frame_count, time_info, status):
    global audio_queue

    # Convert the input data to a numpy array
    audio_data = np.frombuffer(in_data, dtype=np.int16)

    # Check if the condition is met (for example, if the sum of audio exceeds a threshold)
    if PROCESS_AUDIO and is_silent(audio_data):
        print("Processing audio...")
        # Send audio data to the API worker thread for processing
        asyncio.run_coroutine_threadsafe(audio_queue.put(audio_data.tobytes()), loop)

        # Block until the processed audio is available
        processed_audio = asyncio.run_coroutine_threadsafe(audio_queue.get(), loop).result()
    else:
        processed_audio = audio_data.tobytes()

    # Return the processed or original audio for output
    return (processed_audio, pyaudio.paContinue)

def is_silent(audio_data):
    mean_amplitude = np.abs(audio_data).mean()
    print(mean_amplitude)

    if mean_amplitude < SILENCE_THRESHOLD:
        return True 
    return False

def save_audio(block, filename):
    with wave.open(os.path.join(OUTPUT_DIR, filename), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(RATE)
        wf.writeframes(block)
    #print("Audio block saved")

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
    return transcription.text

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open the stream for both input and output
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

# Create an asyncio event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Start the async worker task for API calls
loop.create_task(api_worker())

# Start the audio stream
stream.start_stream()
print("Autotone has started...")

try:
    # Keep the stream running and the asyncio event loop active
    loop.run_forever()
except KeyboardInterrupt:
    pass

# Stop the stream and close the PyAudio session
stream.stop_stream()
stream.close()
p.terminate()

# Close the event loop when done
loop.stop()