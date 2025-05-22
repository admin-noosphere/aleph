#!/usr/bin/env python3
"""
Module de processeur de délai audio pour pipecat.
Introduit un délai configurable pour les frames audio TTSAudioRawFrame.
"""

import asyncio
import logging
from collections import deque
from typing import Optional
import os
import time
import wave

from pipecat.frames.frames import Frame, AudioRawFrame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AudioDelayProcessor(FrameProcessor):
    """
    Un FrameProcessor qui retarde les frames audio (TTSAudioRawFrame ou AudioRawFrame) 
    en direction de la sortie (DOWNSTREAM) d'une durée spécifiée.
    Les autres types de frames sont transmis immédiatement.
    """

    def __init__(self, delay_seconds: float, audio_sample_rate: int = 24000, 
                audio_channels: int = 1, audio_sample_width: int = 2):
        super().__init__(name="audio_delay_processor")
        self._logger = logging.getLogger(__name__)
        self.delay_seconds = delay_seconds
        self.sample_rate = audio_sample_rate
        self.channels = audio_channels
        self.sample_width = audio_sample_width  # Bytes par sample (ex: 2 pour 16-bit PCM)
        
        # Ajouter ces variables dans AudioDelayProcessor aussi
        self.ENABLE_RESAMPLING = False
        self.SAVE_AUDIO = True
        
        self._logger.info(f"AudioDelayProcessor: Resampling {'ACTIVÉ' if self.ENABLE_RESAMPLING else 'DÉSACTIVÉ'}")
        self._logger.info(f"AudioDelayProcessor: Sauvegarde audio {'ACTIVÉE' if self.SAVE_AUDIO else 'DÉSACTIVÉE'}")
        
        # Calculer la taille du buffer en octets pour le délai souhaité
        # Octets par seconde = sample_rate * channels * sample_width
        bytes_per_second = self.sample_rate * self.channels * self.sample_width
        self._delay_buffer_size_bytes = int(bytes_per_second * self.delay_seconds)
        
        self._audio_byte_buffer = bytearray()  # Buffer pour les données audio brutes
        self._delayed_frames_queue = deque()  # Queue pour les frames non-audio qui doivent aussi être retardées relativement
        
        self._current_buffered_audio_duration_ms = 0  # Pour suivre la durée audio actuellement bufferisée

        # Variable qui accumulera tout l'audio d'une phrase TTS
        self._current_tts_buffer = bytearray()
        self._is_tts_speaking = False

        self._logger.info(f"AudioDelayProcessor initialisé avec un délai de {self.delay_seconds}s, "
                         f"correspondant à {self._delay_buffer_size_bytes} octets @ {self.sample_rate}Hz.")

        # Créer le dossier pour les fichiers WAV
        self.wave_dir = os.path.join(os.path.dirname(__file__), "..", "..", "wave_tts_output")
        os.makedirs(self.wave_dir, exist_ok=True)
        self._logger.info(f"Dossier pour les WAV du TTS: {self.wave_dir}")

        # Ajoutez ces variables dans __init__:
        self._buffer_for_saving = bytearray()
        self._last_save_time = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:
            # Par celle-ci (pour ne traiter QUE l'audio TTS):
            is_audio_frame_to_delay = isinstance(frame, TTSAudioRawFrame)

            if is_audio_frame_to_delay and hasattr(frame, 'audio') and frame.audio:
                # Log pour vérifier la fréquence d'échantillonnage réelle
                if hasattr(frame, 'sample_rate'):
                    self._logger.info(f"Frame audio reçu avec sample_rate={frame.sample_rate} Hz")
                
                self._audio_byte_buffer.extend(frame.audio)
                
                # Aussi l'ajouter au buffer pour sauvegarde
                self._buffer_for_saving.extend(frame.audio)
                
                # Calculer la durée actuelle de l'audio dans le buffer
                current_bytes_in_buffer = len(self._audio_byte_buffer)
                bytes_per_sample_frame = self.channels * self.sample_width
                num_samples_in_buffer = current_bytes_in_buffer / bytes_per_sample_frame
                self._current_buffered_audio_duration_ms = (num_samples_in_buffer / self.sample_rate) * 1000

                self._logger.debug(f"Audio ajouté au buffer de délai. Buffer actuel: {current_bytes_in_buffer} bytes / {self._delay_buffer_size_bytes} bytes")

                # Si le buffer de délai est suffisamment rempli (500ms d'audio)
                now = time.time()
                if self.SAVE_AUDIO and len(self._buffer_for_saving) > 0 and (now - self._last_save_time) >= 0.5:
                    filename = os.path.join(self.wave_dir, f"delay_buffer_500ms_{now}.wav")
                    with wave.open(filename, 'wb') as wf:
                        wf.setnchannels(self.channels)
                        wf.setsampwidth(self.sample_width)
                        wf.setframerate(frame.sample_rate if hasattr(frame, 'sample_rate') else self.sample_rate)
                        wf.writeframes(self._buffer_for_saving)
                    self._logger.info(f"Sauvegardé buffer de délai complet de {len(self._buffer_for_saving)} bytes dans {filename}")
                    self._buffer_for_saving = bytearray()
                    self._last_save_time = now

                # Libérer l'audio (et les frames en attente) une fois que le buffer de délai est plein
                while len(self._audio_byte_buffer) >= self._delay_buffer_size_bytes:
                    # Envoyer d'abord les frames non-audio accumulées
                    while self._delayed_frames_queue:
                        await self.push_frame(self._delayed_frames_queue.popleft(), direction)
                    
                    # Envoyer le segment audio qui a atteint le délai - par chunks de 20ms
                    chunk_size_bytes = int(self.sample_rate * self.channels * self.sample_width * 0.020) 
                    
                    if len(self._audio_byte_buffer) > 0:
                        bytes_to_send_from_buffer = min(chunk_size_bytes, len(self._audio_byte_buffer))
                        audio_chunk_to_send = self._audio_byte_buffer[:bytes_to_send_from_buffer]
                        self._audio_byte_buffer = self._audio_byte_buffer[bytes_to_send_from_buffer:]

                        # Recréer un frame audio avec ce chunk - UTILISER LA MÊME FRÉQUENCE D'ÉCHANTILLONNAGE
                        frame_class = type(frame)
                        actual_sample_rate = self.sample_rate if self.ENABLE_RESAMPLING else (frame.sample_rate if hasattr(frame, 'sample_rate') else self.sample_rate)
                        if self.ENABLE_RESAMPLING:
                            self._logger.info(f"AudioDelayProcessor: RESAMPLING à {actual_sample_rate} Hz. (Original: {frame.sample_rate})")
                        else:
                            self._logger.info(f"AudioDelayProcessor: Conservation SR d'origine: {actual_sample_rate} Hz")
                        delayed_audio_frame = frame_class(bytes(audio_chunk_to_send), actual_sample_rate, self.channels)

                        await self.push_frame(delayed_audio_frame, direction)
                        self._logger.debug(f"Envoyé chunk audio retardé: {len(audio_chunk_to_send)} bytes avec sample_rate={actual_sample_rate} Hz")
                
                # Ne pas propager le frame audio original immédiatement car il est bufferisé
                return 
            
            elif isinstance(frame, TTSStartedFrame):
                self._is_tts_speaking = True
                self._current_tts_buffer = bytearray()
                self._delayed_frames_queue.append(frame)
                self._logger.debug(f"Frame {type(frame).__name__} mis en attente dans delayed_frames_queue.")
                return  # Ne pas propager immédiatement

            elif isinstance(frame, TTSStoppedFrame):
                if self.SAVE_AUDIO and len(self._current_tts_buffer) > 0:
                    timestamp = time.time()
                    filename = os.path.join(self.wave_dir, f"tts_complete_{timestamp}.wav")
                    with wave.open(filename, 'wb') as wf:
                        wf.setnchannels(self.channels)
                        wf.setsampwidth(self.sample_width)
                        wf.setframerate(self.sample_rate)
                        wf.writeframes(self._current_tts_buffer)
                    self._logger.info(f"TTS audio: Sauvegardé phrase complète ({len(self._current_tts_buffer)} bytes) dans {filename}")
                self._is_tts_speaking = False
                self._delayed_frames_queue.append(frame)
                self._logger.debug(f"Frame {type(frame).__name__} mis en attente dans delayed_frames_queue.")
                return  # Ne pas propager immédiatement

            else:
                # Pour tous les autres types de frames, on les propage immédiatement
                self._logger.debug(f"Propagation immédiate du frame: {type(frame).__name__}")
                await self.push_frame(frame, direction)

        elif direction == FrameDirection.UPSTREAM:
            # Les frames upstream (venant de l'utilisateur vers le LLM) ne sont pas retardés
            await self.push_frame(frame, direction)

    async def cleanup(self):
        # S'assurer de vider tous les buffers à la fin
        self._logger.info("Cleanup AudioDelayProcessor: envoi des frames audio et de contrôle restants.")
        while self._delayed_frames_queue:
            await self.push_frame(self._delayed_frames_queue.popleft(), FrameDirection.DOWNSTREAM)

        chunk_size_bytes = int(self.sample_rate * self.channels * self.sample_width * 0.020)
        while len(self._audio_byte_buffer) > 0:
            bytes_to_send = min(chunk_size_bytes, len(self._audio_byte_buffer))
            audio_chunk_to_send = self._audio_byte_buffer[:bytes_to_send]
            self._audio_byte_buffer = self._audio_byte_buffer[bytes_to_send:]
            
            # Utiliser TTSAudioRawFrame par défaut pour les chunks restants
            delayed_audio_frame = TTSAudioRawFrame(bytes(audio_chunk_to_send), self.sample_rate, self.channels)
            await self.push_frame(delayed_audio_frame, FrameDirection.DOWNSTREAM)
            self._logger.debug(f"Cleanup: Envoyé chunk audio retardé: {len(audio_chunk_to_send)} bytes.")
        
        self._logger.info("Cleanup AudioDelayProcessor terminé.")
        await super().cleanup()