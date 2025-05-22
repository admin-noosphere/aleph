import logging
import io
import time
import wave
import aiohttp
from pipecat.frames.frames import Frame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

class ApiSenderProcessor(FrameProcessor):
    def __init__(self, api_url, chunk_size_ms=500):
        super().__init__(name="api_sender_processor")
        self.api_url = api_url
        self.chunk_size_ms = chunk_size_ms
        self._buffer = bytearray()
        self._last_send_time = 0
        self._logger = logging.getLogger(__name__)
        self._logger.info(f"ApiSenderProcessor initialisé avec URL: {api_url}, chunk size: {chunk_size_ms}ms")
        
    async def process_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        await super().process_frame(frame, direction)
        
        # Ne traiter que l'audio TTS en direction sortante
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TTSAudioRawFrame):
            if hasattr(frame, 'audio') and frame.audio:
                # Ajouter au buffer
                self._buffer.extend(frame.audio)
                
                # Calculer la taille du buffer pour la durée voulue
                sample_rate = frame.sample_rate if hasattr(frame, 'sample_rate') else 16000
                bytes_per_ms = (sample_rate * 2) / 1000  # 2 bytes par sample (16-bit)
                buffer_size_threshold = int(bytes_per_ms * self.chunk_size_ms)
                
                # Si suffisamment de données ou intervalle de temps atteint
                now = time.time()
                if len(self._buffer) >= buffer_size_threshold or (now - self._last_send_time) >= 0.5:
                    await self.send_to_api(self._buffer, sample_rate)
                    self._buffer = bytearray()
                    self._last_send_time = now
        
        # Toujours transmettre le frame original sans modification
        await self.push_frame(frame, direction)
    
    async def send_to_api(self, audio_data, sample_rate):
        """Envoie l'audio à l'API pour traitement."""
        try:
            # Créer un fichier WAV en mémoire
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit PCM
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)
            
            wav_bytes = wav_buffer.getvalue()
            
            self._logger.info(f"Envoi à l'API: {len(audio_data)} bytes, {sample_rate}Hz à {self.api_url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    data=wav_bytes, 
                    headers={'Content-Type': 'audio/wav'}, 
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        self._logger.info(f"✅ API a traité l'audio (chunk de {self.chunk_size_ms}ms)")
                        return True
                    else:
                        self._logger.error(f"❌ Erreur API: {response.status} - {await response.text()}")
                        return False
        except Exception as e:
            self._logger.error(f"❌ Erreur lors de l'envoi à l'API: {e}")
            return False
