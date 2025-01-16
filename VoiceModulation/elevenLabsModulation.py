import sounddevice as sd
import numpy as np
import wave
import time as timePack
import os
from openai import OpenAI

# Settings
AUDIO_ID = 777
SAMPLE_RATE = 8000
BLOCK_DURATION = 5
BLOCK_SIZE = SAMPLE_RATE * BLOCK_DURATION
CHANNELS = 1
BLOCK_OUTPUT_DIR = os.path.join("audio_blocks", f"P{AUDIO_ID}")
BLOCK_ID = 0