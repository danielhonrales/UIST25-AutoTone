from elevenlabs import ElevenLabs
from getApiKey import get_key
from collections import deque
from openai import OpenAI
import numpy as np
import pyaudio
import asyncio
import wave
import json
import os

# Constants
AUDIO_ID = 777
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 4096
API_URL = "https://your-api-url.com"  # Replace with your actual API URL
DEQUE_SIZE = 200
LISTENER_MAXLEN = 15
TRANSCRIPTION_THRESHOLD = 150  # Threshold for queue size before processing audio
SILENCE_THRESHOLD = 2000  # Example threshold for silence detection (if desired)
OUTPUT_DIR = os.path.join("participants", f"p{AUDIO_ID}")

p = pyaudio.PyAudio()

listener_deque = deque(maxlen=LISTENER_MAXLEN)
transcription_deque = deque(maxlen=DEQUE_SIZE)
analyzer_deque = deque(maxlen=5)
processed_audio_deque = deque(maxlen=DEQUE_SIZE)

openai = OpenAI(
    api_key=get_key("apiKey_openai")
)
elevenlabs = ElevenLabs(
    api_key=get_key("apiKey_elevenlabs")
)
os.makedirs("participants", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

sentiment_score = 0

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

# Async function to handle transcription
async def transcriber():
    while True:
        # If silence or sufficient size
        if (len(transcription_deque) > 0 and is_silent(listener_deque)) or len(transcription_deque) >= TRANSCRIPTION_THRESHOLD:
            audio_data = b''
            for val in transcription_deque:
                audio_data += val
            transcription_deque.clear()

            # Process the audio asynchronously (API call, etc.)
            transcription = await asyncio.to_thread(transcribe_audio, audio_data, AUDIO_ID)
            analyzer_deque.append(transcription)
        else:
            await asyncio.sleep(0.1)  # Sleep a little to avoid busy-waiting

# Function to transcribe audio through API
async def transcribe_audio(audio_data, id):
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

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

async def analyzer():
    global sentiment_score

    while True:
        if len(analyzer_deque) > 0:
            print("Analyzing sentiment")
            transcription = analyzer_deque.popleft()

            #TODO: add transcription validation
            results = await analyze_sentiment(transcription)
            
            # TODO: more complex sentiment tracker, maybe decaying weight over time
            sentiment_score = (results["Sentiment"] + sentiment_score) / 2
            print(f"Global sentiment score: {sentiment_score}")

async def analyze_sentiment(transcription):
    completion = await openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "developer", 
                "content": "You are a barebones analyzer of audio transcipts. You respond with the sentiments and emotions (and associated scores) analyzed from text in JSON format."
            },
            {
                "role": "assistant",
                "content": '{"Sentiment": "Positive (.504)", "Emotions": "[Happy (.952), Hopeful (.75)]"}'
            },
            {
                "role": "user",
                "content": "Analyze the sentiment and emotions in this text: " + transcription
            }
        ]
    )
    
    results = json.loads(completion.choices[0].message.content)

    print(f"Sentiment: {results['Sentiment']}, Emotions: {results['Emotions']}")
    return results

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

async def modulator():
    while True:
        pass

def is_silent(audio_data):
    for chunk in audio_data:
        mean_amplitude = np.abs(chunk).mean()

        if mean_amplitude > SILENCE_THRESHOLD:
            return False 
    return True

def save_audio(block, filename):
    with wave.open(os.path.join(OUTPUT_DIR, filename), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(RATE)
        wf.writeframes(block)
    #print("Audio block saved")

# Callback for the audio stream
def callback(in_data, frame_count, time_info, status):
    # Capture audio data from input
    audio_data = np.frombuffer(in_data, dtype=np.int16)

    listener_deque.append(audio_data)
    # Record audio for processing if not silent or if recent chunks are not silent
    if not is_silent([audio_data]) or not is_silent(listener_deque):
        transcription_deque.append(audio_data.tobytes())

    print(f"Listener: {len(listener_deque)} / {LISTENER_MAXLEN}, Transcriber: {len(transcription_deque)} / {TRANSCRIPTION_THRESHOLD}")

    # Check if processed audio is available
    if len(processed_audio_deque) > 0:
        # If processed audio is available, get it from the queue and output it
        processed_audio = processed_audio_deque.pop()
        return (processed_audio, pyaudio.paContinue)  # Output processed audio
    else:
        # If no processed audio, output the original in_data (input audio)
        return (audio_data, pyaudio.paContinue)  # Output original input audio
    
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

# Open the PyAudio stream for input and output
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

# Start the asyncio event loop
loop = asyncio.get_event_loop()

# Start the audio processing in the background
loop.create_task(transcriber())
#loop.create_task(analyzer())

# Start the PyAudio stream
stream.start_stream()

print("Audio stream is running... Press Ctrl+C to stop.")

# Keep the event loop running
try:
    while stream.is_active():
        loop.run_forever()
except KeyboardInterrupt:
    pass

# Stop the stream after use
stream.stop_stream()
stream.close()
p.terminate()

# Stop the asyncio event loop
loop.stop()