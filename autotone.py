from elevenlabs import ElevenLabs, play, save
from getApiKey import get_key
from collections import deque
from openai import OpenAI
import numpy as np
import pyaudio
import asyncio
import wave
import json
import os
import websockets

# Constants
AUDIO_ID = 777
FORMAT = pyaudio.paInt16
SAMPLE_WIDTH = 2
CHANNELS = 1
RATE = 24000
CHUNK = 512
API_URL = "https://your-api-url.com"  # Replace with your actual API URL
DEQUE_SIZE = 200
LISTENER_MAXLEN = 30
TRANSCRIPTION_THRESHOLD = 500  # Threshold for queue size before processing audio
MODULATION_THRESHOLD = 500 
SILENCE_THRESHOLD = 2000  # Example threshold for silence detection (if desired)
OUTPUT_DIR = os.path.join("participants", f"p{AUDIO_ID}")
VOICE_IDS = {
    "no_mod": "nofx",
    "neutral": "e12e0d3b-0169-4b7c-8a64-a538710f380e",
    "happy": "4c750653-3dcc-4940-989f-35b15ad28ce6",
    "sad": "c23efb14-dbdd-44da-8d32-2ab25956a12c"
}

p = pyaudio.PyAudio()

listener_deque = deque(maxlen=LISTENER_MAXLEN)
transcription_deque = deque(maxlen=DEQUE_SIZE)
analyzer_deque = deque(maxlen=5)
modulation_deque = deque()
processed_audio_deque = deque()

openai = OpenAI(
    api_key=get_key("apiKey_openai")
)
os.makedirs("participants", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

running_sentiment_score = 0
modulating_voice = VOICE_IDS["no_mod"]

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

# Async function to handle transcription
async def transcriber():
    print("Transcriber started...")
    while True:
        # If silence or sufficient size
        if (len(transcription_deque) > 0 and is_silent(listener_deque)) or len(transcription_deque) >= TRANSCRIPTION_THRESHOLD:
            audio_data = b''
            for val in transcription_deque:
                audio_data += val
            transcription_deque.clear()

            # Process the audio asynchronously (API call, etc.)
            transcription = await transcribe_audio(audio_data, AUDIO_ID)
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
        """ with open(os.path.join(OUTPUT_DIR, f"transcript_{id}"), "w+") as transcript_file:
            transcript_file.write(transcription.text) """

    return transcription.text

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

async def analyzer():
    global running_sentiment_score
    global modulating_voice

    print("Analyzer started...")
    running_transcription = ""
    while True:
        if len(analyzer_deque) > 0:
            print("Analyzing sentiment")
            transcription = analyzer_deque.popleft()
            running_transcription = running_transcription + " " + transcription

            #TODO: add transcription validation
            results = await analyze_sentiment(running_transcription)
            
            # TODO: more complex sentiment tracker, maybe decaying weight over time
            sentiment = results["Sentiment"]["name"]
            sentiment_score = results["Sentiment"]["score"]
            if sentiment == "Positive":
                running_sentiment_score = (sentiment_score + running_sentiment_score) / 2
            elif sentiment == "Negative":
                running_sentiment_score = (-sentiment_score + running_sentiment_score) / 2
            else:
                running_sentiment_score = ((sentiment_score / 2) + running_sentiment_score) / 2
            print(f"Running sentiment score: {running_sentiment_score}")
            
            # Inform modulator
            new_voice = select_voice(running_sentiment_score)
            if new_voice != modulating_voice:
                modulating_voice = new_voice
                modulation_deque.append(new_voice)
            
        else:
            await asyncio.sleep(0.1)  # Sleep a little to avoid busy-waiting

async def analyze_sentiment(transcription):
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "developer", 
                "content": """You are a barebones analyzer of audio transcipts. 
                You respond with the sentiments and emotions analyzed from text in JSON format. 
                Provide the sentiment with name and score properties. 
                Provide the emotions as two index-aligned lists for name and score.
                For emotion, provide analysis based on 4 emotions: Happy, Sad, Angry, Surprise.
                Always provide a score for each of the 4 emotions, even if it is 0."""
            },
            {
                "role": "assistant",
                "content": '{"Sentiment": {"name": "Positive", "score": 0.52}, "Emotions": {"names": ["Happy", "Sad", "Angry", "Surprise"], "scores": [".98", ".40", "0", "0.03"]}}'
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
    try:
        async with websockets.connect("ws://localhost:59129/v1") as websocket:
            # Register with API server
            register_message = {
                "id": "ff7d7f15-0cbf-4c44-bc31-b56e0a6c9fa6",
                "action": "registerClient",
                "payload": {
                    "clientKey": get_key("apiKey_voicemod")
                }
            }
            print(f"Connecting with VoiceMod API")
            await websocket.send(json.dumps(register_message))
            registered = False
            while not registered:
                try:
                    response = await websocket.recv()
                    response_json = json.loads(response)
                    includes_msg = False
                    for key in response_json.keys():
                        if key == "msg": includes_msg = True
                    if not includes_msg or response_json["msg"] != "Pending authentication":
                        status_code = response_json["payload"]["status"]["code"]
                        if status_code == 200:
                            registered = True
                        """ else:
                            raise Exception(f"Could not register with VoiceMod, {response}") """
                except Exception as e:
                    raise e
            print(f"Connected to Voicemod API, {response}")

            # TODO: Initialize voice mod
            command = {
                "action": "loadVoice",
                "id": f"load_voice_message_initial",
                "payload": {
                    "voiceID": VOICE_IDS["neutral"]
                }
            }
            print(f"Changing voice to initial neutral")
            await websocket.send(json.dumps(command))
            response = await websocket.recv()
            print(f"Changed voice, {response}")

            # Modulate
            message_id = 0
            print("Modulator started...")
            while True:
                if len(modulation_deque) > 0:
                    voice_profile = modulation_deque.popleft()
                    command = {
                        "action": "loadVoice",
                        "id": f"load_voice_message_{message_id}",
                        "payload": {
                            "voiceID": VOICE_IDS[voice_profile]
                        }
                    }

                    print(f"Changing voice to {voice_profile}")
                    await websocket.send(json.dumps(command))
                    message_id += 1

                    response = await websocket.recv()
                    print(f"Changed voice, {response}")
                else:
                    await asyncio.sleep(0.1)  # Sleep a little to avoid busy-waiting
    except Exception as e:
        print(f"Error with modulator: {e}")

def select_voice(score):
    if -0.3 < score < 0.3:
        return "neutral"
    elif score < -0.3:
        return "sad"
    elif score > 0.3:
        return "happy"

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

def is_silent(audio_data):
    copy = audio_data.copy()
    for chunk in copy:
        mean_amplitude = np.abs(chunk).mean()
        #print(f"Silence Check: {mean_amplitude} / {SILENCE_THRESHOLD}")

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
    try:
        # Capture audio data from input
        audio_data = np.frombuffer(in_data, dtype=np.int16)

        listener_deque.append(audio_data)
        # Record audio for processing if not silent or if recent chunks are not silent
        if not is_silent([audio_data]) or not is_silent(listener_deque):
            transcription_deque.append(audio_data.tobytes())

        #print(f"Listener: {len(listener_deque)} / {LISTENER_MAXLEN}, Transcriber: {len(transcription_deque)} / {TRANSCRIPTION_THRESHOLD}")

        return (None, pyaudio.paContinue)
    except Exception as e:
        print(f"Error with stream, {e}")
    
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

async def main():
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

    # Start the PyAudio stream
    stream.start_stream()
    print("Audio stream is running... Press Ctrl+C to stop.")

    transcriber_task = asyncio.create_task(transcriber())
    analyzer_task = asyncio.create_task(analyzer())
    modulator_task = asyncio.create_task(modulator())

    await transcriber_task
    await analyzer_task
    await modulator_task

    # Stop the stream after use
    stream.stop_stream()
    stream.close()
    p.terminate()

asyncio.run(main())