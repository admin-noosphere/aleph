#!/usr/bin/env python3
"""
Module de traitement audio pour Gala v1
Gère le buffering et le prétraitement audio
"""

import numpy as np
from collections import deque
from typing import Optional


class AudioProcessor:
    """Processeur audio avec buffering"""
    
    def __init__(self, config: dict):
        """
        Initialise le processeur audio
        
        Args:
            config: Configuration audio
        """
        self.sample_rate = config.get("sample_rate", 48000)
        self.channels = config.get("channels", 1)
        self.format = config.get("audio_format", "int16")
        self.min_buffer_ms = config.get("min_audio_ms", 200)
        
        # Buffer pour accumuler l'audio
        self.audio_buffer = deque()
        self.buffer_lock = None  # Threading lock si nécessaire
        
    def get_buffer(self) -> deque:
        """Retourne le buffer audio actuel"""
        return self.audio_buffer
        
    def add_audio(self, audio_data: bytes):
        """Ajoute de l'audio au buffer"""
        self.audio_buffer.append(audio_data)
        
    def get_buffer_duration_ms(self) -> float:
        """Calcule la durée du buffer en ms"""
        total_bytes = sum(len(chunk) for chunk in self.audio_buffer)
        bytes_per_sample = 2 if self.format == "int16" else 4
        samples = total_bytes / (bytes_per_sample * self.channels)
        duration_ms = (samples / self.sample_rate) * 1000
        return duration_ms
        
    def is_buffer_ready(self) -> bool:
        """Vérifie si le buffer a assez d'audio"""
        return self.get_buffer_duration_ms() >= self.min_buffer_ms
        
    def consume_buffer(self) -> Optional[bytes]:
        """
        Consomme le buffer et retourne l'audio combiné
        
        Returns:
            Audio combiné ou None si pas assez
        """
        if not self.is_buffer_ready():
            return None
            
        # Combiner tous les chunks
        combined_audio = b''.join(self.audio_buffer)
        self.audio_buffer.clear()
        
        return combined_audio
        
    def reset(self):
        """Reset le buffer"""
        self.audio_buffer.clear()