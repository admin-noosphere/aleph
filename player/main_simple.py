#!/usr/bin/env python3
"""
Gala - Version simplifiée qui envoie uniquement l'audio à l'API
Pas de gestion LiveLink - tout est géré par l'API
"""

import asyncio
import itertools
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ajouter le chemin src au path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import aiohttp
import google.generativeai as genai
import requests
import wave
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
from pipecat.frames.frames import TTSSpeakFrame, TTSStartedFrame, TTSStoppedFrame, TTSAudioRawFrame
from pipecat.audio.vad.silero import SileroVADAnalyzer

# ---------------------------------------------------------------------------
# Configuration du logging
# ---------------------------------------------------------------------------
logging.basicConfig()
logging.getLogger("pipecat").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Client API minimaliste
# ---------------------------------------------------------------------------
class SimpleApiClient:
    """Client simple pour l'API audio-to-blendshapes"""
    
    def __init__(self, host="127.0.0.1", port=6969):
        self.base_url = f"http://{host}:{port}"
        self.logger = logging.getLogger(__name__)
        
    async def send_audio(self, audio_data: bytes, sample_rate: int = 16000):
        """Envoie l'audio à l'API pour traitement"""
        try:
            async with aiohttp.ClientSession() as session:
                # Envoyer l'audio directement
                async with session.post(
                    f"{self.base_url}/audio_to_blendshapes",
                    data=audio_data,
                    headers={'Content-Type': 'audio/wav'},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"✅ API a traité l'audio")
                        return result
                    else:
                        self.logger.error(f"❌ Erreur API: {response.status}")
                        return None
                
        except Exception as e:
            self.logger.error(f"❌ Erreur: {e}")
            return None

# ---------------------------------------------------------------------------
# Processeur audio simple
# ---------------------------------------------------------------------------
class SimpleAudioProcessor(FrameProcessor):
    """Processeur qui bufferise et envoie l'audio à l'API"""
    
    def __init__(self, api_client: SimpleApiClient):
        super().__init__(name="simple_audio_processor")
        self.api_client = api_client
        self._logger = logging.getLogger(__name__)
        
        # Buffer pour accumulation
        self._buffer = bytearray()
        self._min_buffer_size = 4800  # ~300ms à 16kHz
        
    async def process_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        # Toujours appeler super()
        await super().process_frame(frame, direction)
        
        # Traiter l'audio uniquement
        if (
            direction == FrameDirection.DOWNSTREAM 
            and hasattr(frame, "audio") 
            and frame.audio
        ):
            audio_data = frame.audio
            if isinstance(audio_data, bytes) and len(audio_data) > 0:
                # Ajouter au buffer
                self._buffer.extend(audio_data)
                
                # Si assez de données, envoyer
                if len(self._buffer) >= self._min_buffer_size:
                    self._logger.info(f"⚡ Envoi de {len(self._buffer)} octets à l'API")
                    
                    # Envoyer à l'API
                    await self.api_client.send_audio(
                        bytes(self._buffer), 
                        sample_rate=16000
                    )
                    
                    # Vider le buffer
                    self._buffer.clear()
        
        # Propager le frame
        await self.push_frame(frame, direction)

# ---------------------------------------------------------------------------
# Pipeline principale
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

# Client API simple
api_client = SimpleApiClient(host=NS_HOST, port=NS_PORT)

# Processeur audio
audio_processor = SimpleAudioProcessor(api_client)

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

# Pipeline simplifiée sans buffer
pipeline = Pipeline([
    transport.input(),
    stt,
    context_agg.user(),
    llm,
    tts,
    audio_processor,  # Simple processeur audio
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
        response = requests.get(f"http://{NS_HOST}:{NS_PORT}/health")
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