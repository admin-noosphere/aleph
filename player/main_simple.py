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
from pipecat.frames.frames import ( # Added specific frame imports
    TTSSpeakFrame, 
    TTSStartedFrame, 
    TTSStoppedFrame, 
    TTSAudioRawFrame,
    AudioRawFrame,  # Remplacer AudioFrame par AudioRawFrame
    VADUserStartedSpeakingFrame, 
    VADUserStoppedSpeakingFrame
)
from pipecat.audio.vad.silero import SileroVADAnalyzer
from src.processors.audio_delay import AudioDelayProcessor

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
        self._logger = logging.getLogger(__name__) # Ensure FrameProcessor initializes this logger
        
        # Buffer pour accumulation
        self._buffer = bytearray()

        # VAD attributes
        self._is_speaking = False
        self._silence_start_time = None
        self._idle_triggered = False

        # Configuration from environment or defaults
        self.ORIGINAL_SAMPLE_RATE = int(os.getenv("ORIGINAL_SAMPLE_RATE", "16000")) # Default if not set
        self.TARGET_API_SAMPLE_RATE = int(os.getenv("TARGET_API_SAMPLE_RATE", "88200"))
        self.MIN_AUDIO_BUFFER_MS = int(os.getenv("MIN_AUDIO_BUFFER_MS", "500"))
        self.SILENCE_TIMEOUT_MS = int(os.getenv("SILENCE_TIMEOUT_MS", "1000"))

        # Calculate min_buffer_size based on original sample rate and buffer duration
        # Assumes 1 channel for initial buffer calculation, 2 bytes/sample (16-bit)
        self._min_buffer_size = int(self.ORIGINAL_SAMPLE_RATE * 1 * 2 * (self.MIN_AUDIO_BUFFER_MS / 1000.0))
        self._logger.info(f"Initialized SimpleAudioProcessor: Target API SR={self.TARGET_API_SAMPLE_RATE}, Min Buffer Ms={self.MIN_AUDIO_BUFFER_MS} ({self._min_buffer_size} bytes), Silence Timeout Ms={self.SILENCE_TIMEOUT_MS}")

    async def process_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        # D'abord appeler la méthode parente pour initialiser correctement le processeur
        await super().process_frame(frame, direction)
        
        # Puis votre logique spécifique
        if isinstance(frame, (VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)):
            is_speaking = isinstance(frame, VADUserStartedSpeakingFrame)
            self._logger.info(f"VAD: Voice activity changed. Speaking: {is_speaking}")
            
            if is_speaking and not self._is_speaking:
                self._is_speaking = True
                self._idle_triggered = False
                self._silence_start_time = None
                self._logger.info("User started speaking. Attempting to call /stop_idle.")
                async with aiohttp.ClientSession() as session:
                    try:
                        await session.post(f"{self.api_client.base_url}/stop_idle", timeout=aiohttp.ClientTimeout(total=2))
                        self._logger.info("Successfully called /stop_idle API.")
                    except Exception as e:
                        self._logger.error(f"Error calling /stop_idle API: {e}")
            elif not is_speaking and self._is_speaking:
                self._is_speaking = False
                self._silence_start_time = time.time()
                self._logger.info("User stopped speaking. Silence timer started.")
            
            await self.push_frame(frame, direction) # Forward VAD frame
            return

        if isinstance(frame, AudioRawFrame):  # Remplacer AudioFrame par AudioRawFrame
            if hasattr(frame, "audio") and frame.audio and hasattr(frame, "sample_rate") and hasattr(frame, "num_channels"):
                if self._is_speaking:
                    self._buffer.extend(frame.audio)
                    # self._logger.debug(f"Audio frame received while speaking. Buffer size: {len(self._buffer)}")
                    if len(self._buffer) >= self._min_buffer_size:
                        self._logger.info(f"Audio buffer full for speaking user. Size: {len(self._buffer)}. Processing...")
                        
                        # Ensure frame.sample_rate and frame.num_channels are valid
                        if frame.sample_rate <= 0 or frame.num_channels <=0:
                            self._logger.error(f"Invalid audio frame properties: SR={frame.sample_rate}, Channels={frame.num_channels}. Skipping.")
                            self._buffer.clear() # Clear buffer to prevent processing invalid data
                            await self.push_frame(frame, direction)
                            return

                        resampled_audio = self._resample_audio(
                            audio_data=bytes(self._buffer),
                            orig_sr=frame.sample_rate,
                            target_sr=self.TARGET_API_SAMPLE_RATE,
                            num_channels=frame.num_channels
                        )
                        if resampled_audio:
                            self._logger.info(f"Resampling successful. Sending {len(resampled_audio)} bytes to API at {self.TARGET_API_SAMPLE_RATE} Hz.")
                            await self.api_client.send_audio(
                                resampled_audio,
                                sample_rate=self.TARGET_API_SAMPLE_RATE,
                                channels=1, # Resampled to mono
                                sample_width=2 # 16-bit
                            )
                        else:
                            self._logger.warning("Resampling failed or returned None. No audio sent to API.")
                        self._buffer.clear()
                elif self._silence_start_time and not self._idle_triggered:
                    if (time.time() - self._silence_start_time) * 1000 > self.SILENCE_TIMEOUT_MS:
                        self._logger.info("Silence timeout reached. Attempting to call /start_idle.")
                        async with aiohttp.ClientSession() as session:
                            try:
                                await session.post(f"{self.api_client.base_url}/start_idle", timeout=aiohttp.ClientTimeout(total=2))
                                self._logger.info("Successfully called /start_idle API.")
                                self._idle_triggered = True
                            except Exception as e:
                                self._logger.error(f"Error calling /start_idle API: {e}")
            
            await self.push_frame(frame, direction) # Forward Audio frame
            return
        
        # For any other frame types, just push them
        self._logger.debug(f"Pushing unhandled frame type: {type(frame)}")
        await self.push_frame(frame, direction)

    def _resample_audio(self, audio_data: bytes, orig_sr: int, target_sr: int, num_channels: int) -> Optional[bytes]:
        """Rééchantillonne l'audio à target_sr pour compatibilité NeuroSync et convertit en mono."""
        try:
            self._logger.info(f"Attempting resampling: {len(audio_data)} bytes, SR_orig={orig_sr}, SR_target={target_sr}, Channels={num_channels}")
            
            # Convertir bytes en int16 numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convertir en float32 pour librosa
            audio_float = audio_np.astype(np.float32) / np.iinfo(np.int16).max
            
            # Convertir en mono si nécessaire
            if num_channels > 1:
                self._logger.info(f"Audio has {num_channels} channels. Converting to mono.")
                # librosa.to_mono expects a float waveform, shape (channels, samples) or (samples,) if already mono
                # If audio_float is (samples,), it's already mono. If it's (samples * channels), it needs reshaping.
                # Assuming interleaved audio if multi-channel, e.g., LRLRLR...
                if audio_float.ndim == 1 and num_channels > 1: # Check if it's a flat array needing reshape
                     # Reshape to (samples, channels) then transpose for to_mono which expects (channels, samples)
                    audio_float_reshaped = audio_float.reshape(-1, num_channels).T 
                    audio_mono_float = librosa.to_mono(audio_float_reshaped)
                    self._logger.info(f"Converted to mono. New shape: {audio_mono_float.shape}")
                elif audio_float.ndim == 2 and audio_float.shape[0] == num_channels: # Already (channels, samples)
                    audio_mono_float = librosa.to_mono(audio_float)
                    self._logger.info(f"Input was (channels, samples). Converted to mono. New shape: {audio_mono_float.shape}")
                else: # Already mono or unsupported shape
                    audio_mono_float = audio_float 
                    if num_channels > 1: # Log if we expected stereo but didn't reshape/convert correctly
                        self._logger.warning(f"Multi-channel audio ({num_channels}) not explicitly converted to mono due to array shape {audio_float.shape}. Proceeding as if mono.")
            else: # Already mono
                audio_mono_float = audio_float
                self._logger.info("Audio is already mono.")

            # Rééchantillonnage avec librosa
            if orig_sr == target_sr:
                self._logger.info(f"Original sample rate ({orig_sr}) matches target ({target_sr}). Skipping resampling.")
                resampled_float = audio_mono_float
            else:
                self._logger.info(f"Resampling from {orig_sr}Hz to {target_sr}Hz using kaiser_best.")
                resampled_float = librosa.resample(
                    audio_mono_float, 
                    orig_sr=orig_sr, 
                    target_sr=target_sr,
                    res_type='kaiser_best' # As specified
                )
                self._logger.info(f"Resampling done. New length: {len(resampled_float)}")
            
            # Reconversion en int16
            resampled_int16 = (resampled_float * np.iinfo(np.int16).max).astype(np.int16)
            
            self._logger.info(f"Resampled audio: length={len(resampled_int16)}, min={np.min(resampled_int16) if len(resampled_int16) > 0 else 'N/A'}, "
                             f"max={np.max(resampled_int16) if len(resampled_int16) > 0 else 'N/A'}, target_sr={target_sr}")
            
            return resampled_int16.tobytes()

        except Exception as e:
            self._logger.error(f"Error during resampling: {e}", exc_info=True)
            return None

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

# Paramètres audio
NEUROSYNC_BUFFER_DURATION_SECONDS = 0.5
ACTUAL_TTS_SAMPLE_RATE = 24000  # Fréquence pour OpenAI TTS

# Processeur de délai pour l'audio sortant
audio_delay = AudioDelayProcessor(
    delay_seconds=NEUROSYNC_BUFFER_DURATION_SECONDS,
    audio_sample_rate=ACTUAL_TTS_SAMPLE_RATE,  # Utiliser la bonne fréquence
    audio_channels=1,
    audio_sample_width=2
)

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
pipeline = Pipeline([
    transport.input(),
    stt,
    context_agg.user(),
    llm,
    tts,
    audio_processor,
    audio_delay,                  # Le processeur de délai avec la bonne SR
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