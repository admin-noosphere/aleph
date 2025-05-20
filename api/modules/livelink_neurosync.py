#!/usr/bin/env python3
"""
Module LiveLink pour Gala v1 - Style NeuroSync_Player
Utilise PyLiveLinkFace pour une compatibilité maximale
"""

import socket
import time
from typing import List, Optional
from modules.pylivelinkface import PyLiveLinkFace, FaceBlendShape


class LiveLinkNeuroSync:
    """Client LiveLink utilisant l'approche NeuroSync_Player"""
    
    def __init__(self, udp_ip: str = "127.0.0.1", udp_port: int = 11111, fps: int = 60):
        """
        Initialise le client LiveLink
        
        Args:
            udp_ip: Adresse IP du serveur LiveLink
            udp_port: Port UDP
            fps: FPS cible pour l'animation
        """
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.fps = fps
        
        # Socket UDP
        self.socket = None
        self._create_socket()
        
        # Instance PyLiveLinkFace
        self.py_face = PyLiveLinkFace(name="GalaFace", fps=fps)
        
        # Mapping des indices ARKit standard (68) vers les indices LiveLink (61)
        self.arkit_to_livelink_mapping = self._create_mapping()
    
    def _create_socket(self):
        """Crée et connecte le socket UDP"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.connect((self.udp_ip, self.udp_port))
    
    def _create_mapping(self) -> dict:
        """
        Crée un mapping entre les 68 blendshapes ARKit et les 61 LiveLink
        Certains blendshapes ARKit sont dupliqués ou n'existent pas dans LiveLink
        """
        mapping = {}
        
        # Les 52 premiers sont directs
        for i in range(52):
            mapping[i] = i
        
        # Les suivants (52-67) sont soit dupliqués soit mappés différemment
        # HeadYaw, HeadPitch, HeadRoll (52-54) - pas dans LiveLink standard
        # EyeBlinkLeft/Right dupliqués (54-55) -> map vers 0,7
        mapping[54] = 0  # EyeBlinkLeft duplicate
        mapping[55] = 7  # EyeBlinkRight duplicate
        
        # EyeOpenLeft/Right (56-57) - inverses de EyeBlink
        mapping[56] = 0  # EyeOpenLeft -> inverse de EyeBlinkLeft
        mapping[57] = 7  # EyeOpenRight -> inverse de EyeBlinkRight
        
        # Brow duplicates (58-61)
        mapping[58] = 41  # BrowLeftDown
        mapping[59] = 44  # BrowLeftUp -> BrowOuterUpLeft
        mapping[60] = 42  # BrowRightDown
        mapping[61] = 45  # BrowRightUp -> BrowOuterUpRight
        
        # Emotions (62-67) - pas dans LiveLink standard
        for i in range(62, 68):
            mapping[i] = None
        
        return mapping
    
    def send_blendshapes(self, blendshapes: List[float]):
        """
        Envoie des blendshapes à Unreal via UDP
        
        Args:
            blendshapes: Liste de 68 valeurs ARKit
        """
        if len(blendshapes) != 68:
            raise ValueError(f"Expected 68 blendshapes, got {len(blendshapes)}")
        
        # Convertir de ARKit (68) vers LiveLink (61)
        livelink_values = [0.0] * 61
        
        for arkit_idx, value in enumerate(blendshapes):
            if arkit_idx in self.arkit_to_livelink_mapping:
                livelink_idx = self.arkit_to_livelink_mapping[arkit_idx]
                if livelink_idx is not None:
                    livelink_values[livelink_idx] = value
        
        # Définir les valeurs dans PyLiveLinkFace
        self.py_face.set_blendshapes(livelink_values)
        
        # Encoder et envoyer
        data = self.py_face.encode()
        self.socket.sendall(data)
    
    def send_blendshapes_direct(self, livelink_values: List[float]):
        """
        Envoie directement 61 valeurs LiveLink (sans conversion)
        
        Args:
            livelink_values: Liste de 61 valeurs LiveLink
        """
        if len(livelink_values) != 61:
            raise ValueError(f"Expected 61 LiveLink values, got {len(livelink_values)}")
        
        self.py_face.set_blendshapes(livelink_values)
        data = self.py_face.encode()
        self.socket.sendall(data)
    
    def set_blendshape(self, index: FaceBlendShape, value: float):
        """
        Définit une seule valeur de blendshape
        
        Args:
            index: FaceBlendShape enum
            value: Valeur entre 0.0 et 1.0
        """
        self.py_face.set_blendshape(index.value, value)
    
    def encode_and_get(self) -> bytes:
        """Encode les données actuelles sans les envoyer"""
        return self.py_face.encode()
    
    def send_current(self):
        """Envoie les valeurs actuelles"""
        data = self.py_face.encode()
        self.socket.sendall(data)
    
    def reset(self):
        """Réinitialise tous les blendshapes à 0"""
        self.py_face.reset()
    
    def close(self):
        """Ferme la connexion UDP"""
        if self.socket:
            self.socket.close()


def create_livelink_connection(udp_ip: str = "127.0.0.1", udp_port: int = 11111) -> LiveLinkNeuroSync:
    """
    Factory function pour créer une connexion LiveLink
    
    Args:
        udp_ip: IP du serveur LiveLink
        udp_port: Port UDP
    
    Returns:
        Instance LiveLinkNeuroSync connectée
    """
    return LiveLinkNeuroSync(udp_ip=udp_ip, udp_port=udp_port)