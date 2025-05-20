"""
Processeur pour envoyer les visèmes et blendshapes via LiveLink.
"""

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


# Enum pour les blendshapes (simplifié)
class FaceBlendShape:
    """Enum pour les blendshapes LiveLink"""

    # Eyes
    EyeBlinkLeft = 0
    EyeBlinkRight = 7

    # Jaw and mouth
    JawOpen = 17
    MouthClose = 18
    MouthFunnel = 19
    MouthPucker = 20
    MouthLeft = 21
    MouthRight = 22
    MouthSmileLeft = 23
    MouthSmileRight = 24
    MouthFrownLeft = 25
    MouthFrownRight = 26
    MouthDimpleLeft = 27
    MouthDimpleRight = 28
    MouthStretchLeft = 29
    MouthStretchRight = 30
    MouthRollLower = 31
    MouthRollUpper = 32


class LiveLinkFace:
    """Encodeur du protocole LiveLink face"""

    def __init__(self, name="ARKit", fps=60):
        # Identifiant unique du sujet LiveLink
        self.uuid = f"${str(uuid.uuid1())}"
        self.name = name
        self.fps = fps
        self._version = 6
        self._blend_shapes = [0.0] * 61
        self._denominator = int(self.fps / 60)
        
        # Ajouter facteurs d'échelle
        self._scaling_factor_mouth = 1.1
        self._scaling_factor_eyes = 1.0
        self._scaling_factor_eyebrows = 0.4

    def set_blendshape(self, index, value):
        """Set a blendshape value (0.0-1.0)"""
        self._blend_shapes[index] = max(0.0, min(1.0, value))

    def set_blendshapes(self, values):
        """Set all blendshapes from a list of values"""
        for i, value in enumerate(values):
            if i < len(self._blend_shapes):
                self._blend_shapes[i] = max(0.0, min(1.0, value))

    def reset_blendshapes(self):
        """Reset all blendshapes to 0"""
        self._blend_shapes = [0.0] * 61

    def encode(self):
        """Encode blendshapes to LiveLink binary format"""
        # Format header
        version_packed = struct.pack("<I", self._version)
        uuid_packed = self.uuid.encode("utf-8")
        name_packed = self.name.encode("utf-8")
        name_length_packed = struct.pack("!i", len(self.name))

        # Timecode (simplified)
        now = datetime.now()
        frames = int((now.second * self.fps) + (now.microsecond * self.fps / 1000000))
        sub_frame = 1056964608  # Default value
        frames_packed = struct.pack("!II", frames, sub_frame)

        # Framerate
        frame_rate_packed = struct.pack("!II", self.fps, self._denominator)

        # Blendshapes
        data_packed = struct.pack("!B61f", 61, *self._blend_shapes)

        # Combine all parts
        return (
            version_packed
            + uuid_packed
            + name_length_packed
            + name_packed
            + frames_packed
            + frame_rate_packed
            + data_packed
        )


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
        
        # Stocker la boucle d'événements principale
        self._main_loop = asyncio.get_event_loop()
        
        # Configuration du dossier de logs
        self._log_dir = Path("../../logs/livelink")

        # Nettoyer et recréer le dossier de logs
        if not self._log_dir.exists():
            self._log_dir.mkdir(exist_ok=True, parents=True)

        # Configurer le logger de fichier
        self._file_logger = logging.getLogger("livelink")
        self._file_logger.setLevel(logging.INFO)

        # Créer un nouveau fichier de log avec horodatage
        log_file = (
            self._log_dir / f"livelink_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler.setFormatter(formatter)
        self._file_logger.addHandler(file_handler)

        # Logger également sur la console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self._file_logger.addHandler(console_handler)

        # Socket connection for UDP
        if self._use_udp:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_ip = os.getenv("LIVELINK_IP", udp_ip)
            self._udp_port = int(os.getenv("LIVELINK_PORT", udp_port))
            self._udp_addr = (self._udp_ip, self._udp_port)
            self._file_logger.info(
                f"LiveLinkDataTrack initialisé - UDP {self._udp_ip}:{self._udp_port}"
            )
            try:
                # Envoi d'un paquet vide pour initialiser la connexion
                self._socket.sendto(b"", self._udp_addr)
            except OSError as exc:
                self._file_logger.error(f"Erreur handshake UDP: {exc}")
        else:
            self._file_logger.info("LiveLinkDataTrack initialisé - Daily datatrack")
            
        # Ajouter ces variables pour l'animation de secours
        self._last_update_time = time.time()
        self._idle_animation_active = False
        self._idle_thread = None
        self._running = True
        
        # Démarrer le thread de surveillance
        self._start_idle_monitor()
        
        self._last_values = [0.0] * 61  # Pour l'interpolation
        self._last_frame_time = time.time()
        self._frame_duration = 1.0 / 30.0  # 30 FPS par défaut
    
    def _start_idle_monitor(self):
        """Démarre un thread qui surveille l'activité et active l'animation de secours si nécessaire"""
        self._idle_thread = threading.Thread(target=self._idle_animation_loop)
        self._idle_thread.daemon = True
        self._idle_thread.start()
    
    def _idle_animation_loop(self):
        """Boucle qui vérifie si des données sont reçues et déclenche l'animation de secours si besoin"""
        while self._running:
            current_time = time.time()
            # Si aucune mise à jour depuis 3 secondes, activer l'animation idle
            if current_time - self._last_update_time > 3.0 and not self._idle_animation_active:
                self._idle_animation_active = True
                # Utiliser la boucle stockée au lieu de get_event_loop()
                asyncio.run_coroutine_threadsafe(self._play_idle_animation(), self._main_loop)
            
            time.sleep(1.0)  # Vérifier toutes les secondes
    
    async def _play_idle_animation(self):
        """Joue une animation de respiration et clignements par défaut"""
        self._file_logger.info("⚠️ Animation par défaut activée - aucune donnée reçue")
        
        frame_count = 0
        sin_offset = 0  # Décalage pour les animations sinusoïdales
        
        while self._idle_animation_active:
            # Réinitialiser les blendshapes
            self._face.reset_blendshapes()
            
            # --- ANIMATION VISAGE ---
            # Animation de respiration (JawOpen)
            breath_value = 0.15 + 0.1 * np.sin(sin_offset)
            self._face.set_blendshape(FaceBlendShape.JawOpen, breath_value)
            
            # Animation légère de la mâchoire latérale
            jaw_side = 0.05 * np.sin(sin_offset * 0.7)
            self._face.set_blendshape(FaceBlendShape.JawLeft, max(0, jaw_side))
            self._face.set_blendshape(FaceBlendShape.JawRight, max(0, -jaw_side))
            
            # Animation des lèvres
            lip_value = 0.02 + 0.02 * np.sin(sin_offset * 1.3)
            self._face.set_blendshape(FaceBlendShape.MouthClose, lip_value)
            
            # Clignement
            if frame_count % 60 == 0:  # Toutes les 2 secondes à 30 FPS
                self._face.set_blendshape(FaceBlendShape.EyeBlinkLeft, 0.95)
                self._face.set_blendshape(FaceBlendShape.EyeBlinkRight, 0.95)
            elif frame_count % 60 == 1:  # Ouvrir après 1 frame
                self._face.set_blendshape(FaceBlendShape.EyeBlinkLeft, 0.5)
                self._face.set_blendshape(FaceBlendShape.EyeBlinkRight, 0.5)
            elif frame_count % 60 == 2:  # Ouvrir complètement après 2 frames
                self._face.set_blendshape(FaceBlendShape.EyeBlinkLeft, 0.0)
                self._face.set_blendshape(FaceBlendShape.EyeBlinkRight, 0.0)
            
            # Petit mouvement des sourcils
            brow_value = 0.05 * np.sin(sin_offset * 0.5)
            self._face.set_blendshape(FaceBlendShape.BrowDownLeft, max(0, -brow_value))
            self._face.set_blendshape(FaceBlendShape.BrowDownRight, max(0, -brow_value))
            
            # Encoder et envoyer
            try:
                encoded_data = self._face.encode()
                if self._use_udp:
                    self._socket.sendto(encoded_data, self._udp_addr)
                
                # Log périodique
                if frame_count % 30 == 0:  # Une fois par seconde
                    self._file_logger.info(f"Animation idle: frame {frame_count}, jaw={breath_value:.2f}")
            except Exception as e:
                self._file_logger.error(f"Erreur envoi animation idle: {e}")
            
            # Incrémentations
            frame_count += 1
            sin_offset += 0.075  # Vitesse des animations sinusoïdales
            
            # Pause pour maintenir ~30 FPS
            await asyncio.sleep(1/30)
    
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

    async def process_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        # D'abord, traiter le frame avec la méthode parente
        await super().process_frame(frame, direction)
        
        # Ne pas traiter les visèmes ou blendshapes - ils sont traités directement
        # via send_viseme et send_blendshape
        
        # Simplement propager tous les frames (sauf les Viseme/Blendshape que nous ne recevrons plus)
        await self.push_frame(frame, direction)

    def close(self):
        """Nettoie les ressources à la fermeture"""
        self._running = False
        if self._idle_thread:
            self._idle_thread.join(timeout=1.0)
        if hasattr(self, '_socket') and self._socket:
            self._socket.close()
