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
import io # Ajout pour BytesIO
import numpy as np
import librosa
from logging.handlers import RotatingFileHandler

# Ajouter le chemin src au path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import aiohttp
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
from pipecat.frames.frames import TTSSpeakFrame, TTSStartedFrame, TTSStoppedFrame, TTSAudioRawFrame
from pipecat.audio.vad.silero import SileroVADAnalyzer

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
class SimpleApiClient:
    """Client simple pour l'API audio-to-blendshapes"""
    
    def __init__(self, host="127.0.0.1", port=6969):
        self.base_url = f"http://{host}:{port}"
        self.logger = logging.getLogger(__name__)
        
    async def send_audio(self, audio_data: bytes, sample_rate: int = 88200, channels: int = 1, sample_width: int = 2): # Changé 16000 en 88200 par défaut
        """Envoie l'audio à l'API pour traitement après l'avoir encapsulé dans un format WAV."""
        try:
            # Créer un fichier WAV en mémoire
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width) # 2 bytes pour 16-bit PCM
                wf.setframerate(sample_rate)  # Utilisera la valeur de sample_rate (maintenant 88200 Hz par défaut ou passée)
                wf.writeframes(audio_data)
            
            wav_bytes = wav_buffer.getvalue()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/audio_to_blendshapes",
                    data=wav_bytes, 
                    headers={'Content-Type': 'audio/wav'}, 
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"✅ API a traité l'audio (envoyé comme WAV à {sample_rate} Hz)")
                        return result
                    else:
                        self.logger.error(f"❌ Erreur API: {response.status} - {await response.text()}")
                        return None
                
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de la création/envoi du WAV: {e}")
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
        # Envoyer des chunks plus petits pourrait être préférable pour la latence,
        # mais assurez-vous que NeuroSync peut les gérer.
        # 4800 bytes @ 16kHz, 16-bit mono = 150ms.
        # La fonction NeuroSync traite probablement des segments un peu plus longs.
        # Si vous avez toujours des problèmes, essayez d'augmenter cette taille
        # ou d'ajuster la logique de chunking côté API si possible.
        self._min_buffer_size = int(16000 * 2 * 0.5)  #  0.5 mini ~300ms à 16kHz, 16-bit mono (9600 bytes)
                                         # Augmenté pour potentiellement avoir des segments plus stables pour NeuroSync
        
    async def process_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        await super().process_frame(frame, direction)
        
        if (
            direction == FrameDirection.DOWNSTREAM 
            and hasattr(frame, "audio") 
            and frame.audio
        ):
            audio_data = frame.audio
            if isinstance(audio_data, bytes) and len(audio_data) > 0:
                self._buffer.extend(audio_data)
                
                if len(self._buffer) >= self._min_buffer_size:
                    self._logger.info(f"⚡ Envoi de {len(self._buffer)} octets (PCM) à l'API sous format WAV")
                    
                    # !! IMPORTANT !!
                    # Les données dans self._buffer sont probablement à 16000 Hz.
                    # Vous DEVEZ les rééchantillonner à 88200 Hz avant cette étape.
                    # Voir la section sur le rééchantillonnage ci-dessous.
                    # Pour l'instant, nous passons la nouvelle fréquence cible :

                    audio_to_send = bytes(self._buffer) 
                    # Exemple conceptuel de rééchantillonnage (nécessite une implémentation réelle)
                    # resampled_audio_data = await self.resample_audio(audio_to_send, 16000, 88200)

                    await self.api_client.send_audio(
                        # resampled_audio_data, # Idéalement, vous envoyez les données rééchantillonnées
                        audio_to_send,      # Si vous n'avez pas encore le rééchantillonnage, ceci est incorrect pour le modèle
                        sample_rate=88200,  # Changé 16000 en 88200
                        channels=1,
                        sample_width=2 
                    )
                    
                    self._buffer.clear()
        
        await self.push_frame(frame, direction)

    def _resample_audio(self, audio_data):
        """Rééchantillonne l'audio à 88200Hz pour compatibilité NeuroSync."""
        try:
            # Convertir bytes en int16
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Log avant rééchantillonnage 
            self._logger.info(f"Audio original: longueur={len(audio_np)}, min={np.min(audio_np)}, "
                             f"max={np.max(audio_np)}, moyenne={np.mean(np.abs(audio_np))}")
            
            # Normalisation en float32 dans [-1, 1]
            audio_np = audio_np.astype(np.float32) / np.iinfo(np.int16).max
            
            # Rééchantillonnage avec librosa
            resampled_np = librosa.resample(
                audio_np, 
                orig_sr=self._sample_rate, 
                target_sr=88200
            )
            
            # Reconversion en int16
            resampled_int16 = (resampled_np * np.iinfo(np.int16).max).astype(np.int16)
            
            # Log après rééchantillonnage
            self._logger.info(f"Audio rééchantillonné: longueur={len(resampled_int16)}, min={np.min(resampled_int16)}, "
                             f"max={np.max(resampled_int16)}, moyenne={np.mean(np.abs(resampled_int16))}")
            
            # Sauvegarder l'audio pour débogage (optionnel)
            # import wave
            # with wave.open("debug_audio.wav", "wb") as wf:
            #     wf.setnchannels(1)
            #     wf.setsampwidth(2)  # 16 bits
            #     wf.setframerate(88200)
            #     wf.writeframes(resampled_int16.tobytes())
            
            return resampled_int16.tobytes()
        except Exception as e:
            self._logger.error(f"Erreur de rééchantillonnage: {e}")
            return audio_data  # Renvoyer les données d'origine en cas d'erreur

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

# Client API simple
api_client = SimpleApiClient(host=NS_HOST, port=NS_PORT) #

# Processeur audio
audio_processor = SimpleAudioProcessor(api_client) #

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
    audio_processor,  # Simple processeur audio #
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