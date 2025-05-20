from __future__ import annotations

import os
import socket
import json
import uuid
from pathlib import Path
from datetime import datetime
import struct
import numpy as np
import shutil
import logging
import time
import threading
import asyncio

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from src.gala.neurosync_buffer import VisemeFrame, BlendshapeFrame

# Importer les classes de livelink
from src.gala.livelink.connect.pylivelinkface import PyLiveLinkFace
from src.gala.livelink.connect.faceblendshapes import FaceBlendShape

class LiveLinkDataTrack(FrameProcessor):
    """Processeur qui transmet les visèmes et blendshapes vers Unreal via LiveLink."""

    def __init__(
        self, daily_transport=None, use_udp=True, udp_ip="192.168.1.14", udp_port=11111
    ):
        super().__init__(name="livelink_datatrack")
        self._tx = daily_transport
        self._use_udp = use_udp
        self._face = PyLiveLinkFace(name="ARKit")
        self._last_viseme = -1
        self._last_values = [0.0] * 61
        self._last_frame_time = time.time()
        self._frame_duration = 1.0 / 30.0  # 30 FPS

        # Socket connection for UDP
        if self._use_udp:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_ip = os.getenv("LIVELINK_IP", udp_ip)
            self._udp_port = int(os.getenv("LIVELINK_PORT", udp_port))
            self._udp_addr = (self._udp_ip, self._udp_port)
            logging.info(f"LiveLinkDataTrack initialisé - UDP {self._udp_ip}:{self._udp_port}")
            try:
                self._socket.sendto(b"", self._udp_addr)
            except OSError as exc:
                logging.error(f"Erreur handshake UDP: {exc}")
        else:
            logging.info("LiveLinkDataTrack initialisé - Daily datatrack")

    async def send_blendshape(self, frame):
        try:
            # Contrôle du timing pour un envoi régulier
            current_time = time.time()
            elapsed = current_time - self._last_frame_time
            if elapsed < self._frame_duration:
                await asyncio.sleep(self._frame_duration - elapsed)
            self._last_frame_time = time.time()
            
            # Réinitialiser les blendshapes
            for i in range(51):  # ARKit utilise 51 blendshapes standard
                self._face.set_blendshape(FaceBlendShape(i), 0.0)
            
            # Obtenir et traiter les valeurs
            if hasattr(frame, 'values') and frame.values:
                # Normalisation de l'indice 65 (Neutral) qui arrive avec des valeurs ~100
                if len(frame.values) > 65 and frame.values[65] > 1.0:
                    neutral_value = frame.values[65] / 100.0
                    # Utiliser la valeur neutre pour contrôler d'autres blendshapes
                    if neutral_value > 0.5:
                        self._face.set_blendshape(FaceBlendShape(65), neutral_value)  # Neutral
                
                # Traiter les autres blendshapes (limiter à 61 - nombre max supporté par ARKit)
                for i, value in enumerate(frame.values[:61]):
                    # Ignorer les valeurs trop petites
                    if value > 0.05:
                        # Interpolation pour plus de fluidité
                        if i < len(self._last_values):
                            value = 0.7 * value + 0.3 * self._last_values[i]
                        
                        # Appliquer la valeur
                        try:
                            self._face.set_blendshape(FaceBlendShape(i), value)
                        except ValueError:
                            # Ignorer les index non définis dans l'enum
                            pass
            
            # Enregistrer les valeurs pour la prochaine frame
            self._last_values = frame.values[:61] if len(frame.values) >= 61 else frame.values
            
            # Encoder pour LiveLink
            encoded_data = self._face.encode()
            
            # Envoi via UDP
            if self._use_udp:
                self._socket.sendto(encoded_data, self._udp_addr)
            
            # Log périodique
            if hasattr(frame, 'frame_idx') and frame.frame_idx % 10 == 0:
                logging.info(f"LiveLink: envoi frame blendshapes {frame.frame_idx}")
            
            return True
        except Exception as e:
            logging.error(f"Erreur LiveLink blendshape: {e}")
            return False
        
    async def send_viseme(self, frame):
        try:
            viseme_id = frame.id
            time_ms = frame.time_ms

            # Éviter les répétitions inutiles
            if viseme_id == self._last_viseme:
                return True
            
            self._last_viseme = viseme_id

            # Réinitialiser les blendshapes
            for i in range(51):
                self._face.set_blendshape(FaceBlendShape(i), 0.0)

            # Configurer les blendshapes en fonction du visème
            if viseme_id == 0:  # Silence/neutre
                self._face.set_blendshape(FaceBlendShape.JawOpen, 0.1)
            elif viseme_id == 1:  # "aa"
                self._face.set_blendshape(FaceBlendShape.JawOpen, 0.5)
                self._face.set_blendshape(FaceBlendShape.MouthFunnel, 0.6)
            elif viseme_id == 2:  # "ah"
                self._face.set_blendshape(FaceBlendShape.JawOpen, 0.4)
                self._face.set_blendshape(FaceBlendShape.MouthFunnel, 0.4)
            elif viseme_id == 3:  # "oo"
                self._face.set_blendshape(FaceBlendShape.JawOpen, 0.2)
                self._face.set_blendshape(FaceBlendShape.MouthPucker, 0.6)

            # Encoder pour LiveLink
            encoded_data = self._face.encode()

            if self._use_udp:
                # Envoi via UDP
                self._socket.sendto(encoded_data, self._udp_addr)

            logging.info(f"LiveLink: envoi visème {viseme_id} à t={time_ms}ms")
            return True
        
        except Exception as e:
            logging.error(f"Erreur LiveLink viseme: {e}")
            return False

    # Ajoutez ici le reste de vos méthodes adaptées
