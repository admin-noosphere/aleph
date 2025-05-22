import asyncio
import asyncio
import itertools
import json
import logging
import os
import re
import sys
import time # Added import
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import io # Ajout pour BytesIO
import numpy as np
import librosa
import aiohttp # Added import
from logging.handlers import RotatingFileHandler
import wave 

# Ajouter le chemin src au path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import google.generativeai as genai
import requests
import wave # Ajout pour la création de l'en-tête WAV
from dotenv import load_dotenv
from loguru import logger
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.google.llm_openai import GoogleLLMOpenAIBetaService
from pipecat.transports.services.daily import (
    DailyParams,
    DailyTransport,
)
from pipecat.frames.frames import (
    TTSSpeakFrame, 
    TTSStartedFrame, 
    TTSStoppedFrame, 
    TTSAudioRawFrame,
    AudioRawFrame, # Au lieu de AudioFrame
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame
)
from pipecat.audio.vad.silero import SileroVADAnalyzer
from src.processors.audio_delay import AudioDelayProcessor
from src.processors.api_sender import ApiSenderProcessor

# Créer un dossier pour les logs
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)

# Configuration du logger principal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajouter un handler pour fichier
file_handler = RotatingFileHandler(
    os.path.join(log_dir, "player.log"), 
    maxBytes=10*1024*1024,  # 10 Mo maximum
    backupCount=5           # Garder 5 fichiers de sauvegarde
)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# ---------------------------------------------------------------------------
# Configuration du logging
# ---------------------------------------------------------------------------
logging.basicConfig()
logging.getLogger("pipecat").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Client API minimaliste
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pipeline principale (le reste du fichier reste identique)
# ---------------------------------------------------------------------------

# Chargement des variables
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "dbab4e489478bd4338ad6cbb3901a433550d7cf1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
DAILY_ROOM_URL = os.getenv("DAILY_ROOM_URL")
DAILY_API_TOKEN = os.getenv("DAILY_API_TOKEN")
BOT_NAME = os.getenv("BOT_NAME", "Gala")
TTS_SERVICE = os.getenv("TTS_SERVICE", "openai").lower()
NS_HOST = os.getenv("NS_HOST", "127.0.0.1")
NS_PORT = int(os.getenv("NS_PORT", "6969"))

# Services
stt = OpenAISTTService(
    api_key=OPENAI_API_KEY,
    language="fr"
)

if TTS_SERVICE == "elevenlabs":
    tts = ElevenLabsTTSService(
        api_key=ELEVENLABS_API_KEY,
        model="eleven_turbo_v2",
        language="fr"
    )
else:
    tts = OpenAITTSService(
        api_key=OPENAI_API_KEY,
        voice="nova"
    )

llm = GoogleLLMOpenAIBetaService(
    api_key=GEMINI_API_KEY,
    model=GEMINI_MODEL
)

# Transport Daily
UDP_CONNECTIONS = True

transport = DailyTransport(
    DAILY_ROOM_URL,
    DAILY_API_TOKEN,
    BOT_NAME,
    DailyParams(
        audio_out_enabled=True,
        transcription_enabled=False,
        vad_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
        use_udp_connections=UDP_CONNECTIONS
    )
)



# Paramètres audio
NEUROSYNC_BUFFER_DURATION_SECONDS = 0.5
ACTUAL_TTS_SAMPLE_RATE = 16000  # Fréquence pour OpenAI TTS

# Processeur de délai pour l'audio sortant
audio_delay = AudioDelayProcessor(
    delay_seconds=NEUROSYNC_BUFFER_DURATION_SECONDS,
    audio_sample_rate=ACTUAL_TTS_SAMPLE_RATE,
    audio_channels=1,
    audio_sample_width=2
)

# Configuration supplémentaire après l'initialisation
audio_delay.ENABLE_RESAMPLING = False
audio_delay.SAVE_AUDIO = True

# Créer un dossier pour les fichiers WAV
import os
audio_delay.wave_dir = os.path.join(os.path.dirname(__file__), "wave_tts_output")
os.makedirs(audio_delay.wave_dir, exist_ok=True)
logger.info(f"Dossier pour les WAV du TTS: {audio_delay.wave_dir}")

# Messages système
messages = [
    {
        "role": "system",
        "content": """Tu es Gala, une femme pirate IA. Tu es le capitaine de ton navire.
Tu parles toujours en français avec l'accent et l'attitude typique d'un pirate.
Sois fière, audacieuse et aventureuse. Tu es toujours prête pour l'action."""
    }
]

context = OpenAILLMContext(messages)
context_agg = llm.create_context_aggregator(context)

# Définition de la pipeline
api_sender = ApiSenderProcessor(
    api_url=f"http://{NS_HOST}:{NS_PORT}/audio_to_blendshapes"
)

pipeline = Pipeline([
    transport.input(),
    stt,
    context_agg.user(),
    llm,
    tts,
    audio_delay,
    api_sender,        # Nouveau processor dédié à l'envoi API
    context_agg.assistant(),
    transport.output()
])

task = PipelineTask(
    pipeline,
    params=PipelineParams(
        allow_interruptions=True,
        enable_metrics=True,
        enable_usage_metrics=True
    )
)

# Handlers Daily
@transport.event_handler("on_client_connected")
async def on_client_connected(transport, client):
    logger.info("Client connecté: %s", client.get("id"))
    
    greeting = "Ahoy moussaillon! Je suis Gala, capitaine redoutable des sept mers!"
    
    assistant_ctx = context_agg.assistant()
    assistant_ctx.add_messages([{"role": "assistant", "content": greeting}])
    
    await task.queue_frames([TTSSpeakFrame(greeting)])

@transport.event_handler("on_client_disconnected")
async def on_client_disconnected(transport, client):
    logger.info("Client déconnecté: %s", client.get("id"))
    await task.cancel()

# Main
async def main():
    logger.info("Gala • Démarrage – salle %s", DAILY_ROOM_URL)
    
    # Vérifier l'API
    try:
        response = requests.get(f"http://{NS_HOST}:{NS_PORT}/health") #
        if response.status_code == 200:
            logger.info("✅ API accessible")
        else:
            logger.warning("⚠️ API non accessible")
    except:
        logger.warning("⚠️ Impossible de vérifier l'API")
    
    runner = PipelineRunner()
    await runner.run(task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Arrêt par l'utilisateur")
