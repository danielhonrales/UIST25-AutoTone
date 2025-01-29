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

# Constants
AUDIO_ID = 777
FORMAT = pyaudio.paInt16
SAMPLE_WIDTH = 2
CHANNELS = 1
RATE = 24000
CHUNK = 512
API_URL = "https://your-api-url.com"  # Replace with your actual API URL
DEQUE_SIZE = 200
LISTENER_MAXLEN = 15
TRANSCRIPTION_THRESHOLD = 150  # Threshold for queue size before processing audio
MODULATION_THRESHOLD = 150 
SILENCE_THRESHOLD = 2000  # Example threshold for silence detection (if desired)
OUTPUT_DIR = os.path.join("participants", f"p{AUDIO_ID}")
VOICE_IDS = {
    "no_mod": -1,
    "neutral": "goulD9M4G4gdl3jk9hcH",
    "happy": "gJt2pwZO3mYj05aHAJCF",
    "sad": "MHXQNsZO57D7LmsedS4u"
}

p = pyaudio.PyAudio()

listener_deque = deque(maxlen=LISTENER_MAXLEN)
transcription_deque = deque(maxlen=DEQUE_SIZE)
analyzer_deque = deque(maxlen=5)
modulation_deque = deque(maxlen=DEQUE_SIZE)
processed_audio_deque = deque()

openai = OpenAI(
    api_key=get_key("apiKey_openai")
)
elevenlabs = ElevenLabs(
    api_key=get_key("apiKey_elevenlabs")
)
os.makedirs("participants", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

running_sentiment_score = 0
modulating_voice_id = VOICE_IDS["no_mod"]

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
        with open(os.path.join(OUTPUT_DIR, f"transcript_{id}"), "w+") as transcript_file:
            transcript_file.write(transcription.text)

    return transcription.text

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

async def analyzer():
    global running_sentiment_score

    print("Analyzer started...")
    while True:
        if len(analyzer_deque) > 0:
            print("Analyzing sentiment")
            transcription = analyzer_deque.popleft()

            #TODO: add transcription validation
            results = await analyze_sentiment(transcription)
            
            # TODO: more complex sentiment tracker, maybe decaying weight over time
            sentiment_score = results["Sentiment"]["score"]
            running_sentiment_score = (sentiment_score + running_sentiment_score) / 2
            print(f"Running sentiment score: {running_sentiment_score}")
        else:
            await asyncio.sleep(0.1)  # Sleep a little to avoid busy-waiting

async def analyze_sentiment(transcription):
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "developer", 
                "content": "You are a barebones analyzer of audio transcipts. You respond with the sentiments and emotions analyzed from text in JSON format. Provide the sentiment with name and score properties. Prove the emotions as two index-aligned lists for name and score."
            },
            {
                "role": "assistant",
                "content": '{"Sentiment": {"name": "Positive", "score": 0.52}, "Emotions": {"names": ["Happy", "Hopeful"], "scores": [".98", ".75"]}}'
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
    global running_sentiment_score
    global modulating_voice_id
    sentiment_score = running_sentiment_score
    
    print("Modulator started...")
    while True:
        # Select voice when sentiment score changes
        if sentiment_score != running_sentiment_score:
            sentiment_score = running_sentiment_score
            modulating_voice_id = select_voice(sentiment_score)

        # Modulate voice
        if modulating_voice_id != -1:
            if (len(modulation_deque) > 0 and is_silent(listener_deque)) or len(modulation_deque) >= MODULATION_THRESHOLD:
                audio_data = b''
                for val in modulation_deque:
                    audio_data += val
                modulation_deque.clear()

                save_audio(audio_data, f"audio_to_modulate.wav")
                print("Sending audio to elevenlabs")
                with open(os.path.join(OUTPUT_DIR, "audio_to_modulate.wav"), "rb") as audio_file:
                    response = elevenlabs.speech_to_speech.convert(
                        voice_id=modulating_voice_id,
                        enable_logging=True,
                        output_format="pcm_24000",
                        audio=audio_file
                    )

                    for val in response:
                        processed_audio_deque.append(val)

                    print("Received modulated audio")
            else:
                await asyncio.sleep(0.1)
        else:
            await asyncio.sleep(0.1)

def select_voice(score):
    if -0.3 < score < 0.3:
        return VOICE_IDS["no_mod"]
    elif score < -0.3:
        return VOICE_IDS["sad"]
    elif score > 0.3:
        return VOICE_IDS["happy"]


#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

def is_silent(audio_data):
    for chunk in audio_data:
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

            if modulating_voice_id != VOICE_IDS["no_mod"]:
                modulation_deque.append(audio_data.tobytes())

        #print(f"Listener: {len(listener_deque)} / {LISTENER_MAXLEN}, Transcriber: {len(transcription_deque)} / {TRANSCRIPTION_THRESHOLD}")

        # Check if processed audio is available
        if len(processed_audio_deque) > 0:
            # If processed audio is available, get it from the queue and output it
            processed_audio = processed_audio_deque.popleft()

            # If not right size, do silence
            if len(processed_audio) != frame_count * SAMPLE_WIDTH * CHANNELS:
                silence = b'\0' * (frame_count * SAMPLE_WIDTH * CHANNELS)
                return (silence, pyaudio.paContinue)
            
            return (processed_audio, pyaudio.paContinue)  # Output processed audio
            
        else:
            # If no processed audio, output the original in_data (input audio)
            return (audio_data, pyaudio.paContinue)  # Output original input audio
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
                output=True,
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