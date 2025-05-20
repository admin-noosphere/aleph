#!/usr/bin/env python3
"""
Module de processeur de délai audio pour pipecat.
Introduit un délai configurable pour les frames audio TTSAudioRawFrame.
"""

import asyncio
import logging
from collections import deque
from typing import Optional

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
        
        # Calculer la taille du buffer en octets pour le délai souhaité
        # Octets par seconde = sample_rate * channels * sample_width
        bytes_per_second = self.sample_rate * self.channels * self.sample_width
        self._delay_buffer_size_bytes = int(bytes_per_second * self.delay_seconds)
        
        self._audio_byte_buffer = bytearray()  # Buffer pour les données audio brutes
        self._delayed_frames_queue = deque()  # Queue pour les frames non-audio qui doivent aussi être retardées relativement
        
        self._current_buffered_audio_duration_ms = 0  # Pour suivre la durée audio actuellement bufferisée

        self._logger.info(f"AudioDelayProcessor initialisé avec un délai de {self.delay_seconds}s, "
                         f"correspondant à {self._delay_buffer_size_bytes} octets @ {self.sample_rate}Hz.")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:
            # On ne retarde que l'audio du bot (TTS) allant vers la sortie Daily.
            is_audio_frame_to_delay = isinstance(frame, TTSAudioRawFrame) or isinstance(frame, AudioRawFrame)

            if is_audio_frame_to_delay and hasattr(frame, 'audio') and frame.audio:
                # Log pour vérifier la fréquence d'échantillonnage réelle
                if hasattr(frame, 'sample_rate'):
                    self._logger.info(f"Frame audio reçu avec sample_rate={frame.sample_rate} Hz")
                
                self._audio_byte_buffer.extend(frame.audio)
                
                # Calculer la durée actuelle de l'audio dans le buffer
                current_bytes_in_buffer = len(self._audio_byte_buffer)
                bytes_per_sample_frame = self.channels * self.sample_width
                num_samples_in_buffer = current_bytes_in_buffer / bytes_per_sample_frame
                self._current_buffered_audio_duration_ms = (num_samples_in_buffer / self.sample_rate) * 1000

                self._logger.debug(f"Audio ajouté au buffer de délai. Buffer actuel: {current_bytes_in_buffer} bytes / {self._delay_buffer_size_bytes} bytes")

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
                        # actual_sample_rate = frame.sample_rate if hasattr(frame, 'sample_rate') else self.sample_rate # Ligne originale
                        actual_sample_rate = self.sample_rate # FORCER à utiliser la SR de l'initialisation du processeur
                        self._logger.info(f"AudioDelayProcessor: FORCAGE SR de sortie à {actual_sample_rate} Hz. (SR déclarée du frame entrant était: {frame.sample_rate if hasattr(frame, 'sample_rate') else 'N/A'})")
                        delayed_audio_frame = frame_class(bytes(audio_chunk_to_send), actual_sample_rate, self.channels)
                        await self.push_frame(delayed_audio_frame, direction)
                        self._logger.debug(f"Envoyé chunk audio retardé: {len(audio_chunk_to_send)} bytes avec sample_rate={actual_sample_rate} Hz")
                
                # Ne pas propager le frame audio original immédiatement car il est bufferisé
                return 
            
            elif isinstance(frame, (TTSStartedFrame, TTSStoppedFrame)):
                # Ces frames marquent le début et la fin de la parole du TTS.
                # Il faut les retarder de la même manière que l'audio.
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