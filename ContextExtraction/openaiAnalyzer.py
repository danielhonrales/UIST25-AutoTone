from openai import OpenAI
from getApiKey import get_key
import os

# Settings
AUDIO_ID = 777
TRANSCRIPT_OUT_DIR = os.path.join("audio_transcripts", f"P{AUDIO_ID}")

openai = OpenAI(
    api_key=get_key()
)

with open(os.path.join(TRANSCRIPT_OUT_DIR, "transcript"), "r") as transcript_file:
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "developer", 
                "content": "You are a barebones analyzer of audio transcipts. You respond with the sentiments and emotions analyzed from text in JSON format."
            },
            {
                "role": "assistant",
                "content": '{"Sentiment": "Positive (.504)", "Emotions": "Happy (.952), Hopeful (.75)"}'
            },
            {
                "role": "user",
                "content": "Analyze the sentiment and emotions in this text: " + transcript_file.read()
            }
        ]
    )
    
    print(completion.choices[0].message.content)