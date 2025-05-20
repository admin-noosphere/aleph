#!/usr/bin/env python3
"""
Module LiveLink client pour Gala v1
Envoie les blendshapes à Unreal Engine via LiveLink
"""

import asyncio
import socket
import struct
import time
from typing import List, Dict, Optional
import websockets
import json


class LiveLinkClient:
    """Client LiveLink pour Unreal Engine"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 11111, fps: int = 60):
        """
        Initialise le client LiveLink
        
        Args:
            host: Adresse IP du serveur LiveLink
            port: Port LiveLink
            fps: FPS cible
        """
        self.host = host
        self.port = port
        self.fps = fps
        self.frame_time = 1.0 / fps
        
        # WebSocket connection
        self.websocket = None
        self.is_connected = False
        
        # UDP socket pour mode direct
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Noms des blendshapes ARKit
        self.blendshape_names = self._get_arkit_blendshape_names()
        
    def _get_arkit_blendshape_names(self) -> List[str]:
        """Retourne la liste des 68 blendshapes ARKit"""
        return [
            "EyeBlink_L", "EyeBlink_R", "EyeLookDown_L", "EyeLookDown_R",
            "EyeLookIn_L", "EyeLookIn_R", "EyeLookOut_L", "EyeLookOut_R",
            "EyeLookUp_L", "EyeLookUp_R", "EyeSquint_L", "EyeSquint_R",
            "EyeWide_L", "EyeWide_R", "JawForward", "JawLeft", "JawRight",
            "JawOpen", "MouthClose", "MouthFunnel", "MouthPucker", "MouthLeft",
            "MouthRight", "MouthSmile_L", "MouthSmile_R", "MouthFrown_L",
            "MouthFrown_R", "MouthDimple_L", "MouthDimple_R", "MouthStretch_L",
            "MouthStretch_R", "MouthRollLower", "MouthRollUpper", "MouthShrugLower",
            "MouthShrugUpper", "MouthPress_L", "MouthPress_R", "MouthLowerDown_L",
            "MouthLowerDown_R", "MouthUpperUp_L", "MouthUpperUp_R", "BrowDown_L",
            "BrowDown_R", "BrowInnerUp", "BrowOuterUp_L", "BrowOuterUp_R",
            "CheekPuff", "CheekSquint_L", "CheekSquint_R", "NoseSneer_L",
            "NoseSneer_R", "TongueOut", "HeadYaw", "HeadPitch", "HeadRoll",
            "EyeBlinkLeft", "EyeBlinkRight", "EyeOpenLeft", "EyeOpenRight",
            "BrowLeftDown", "BrowLeftUp", "BrowRightDown", "BrowRightUp",
            "EmotionHappy", "EmotionSad", "EmotionAngry", "EmotionSurprised"
        ]
        
    async def connect_websocket(self):
        """Connecte au serveur LiveLink via WebSocket"""
        try:
            uri = f"ws://{self.host}:{self.port}"
            self.websocket = await websockets.connect(uri)
            self.is_connected = True
            print(f"Connected to LiveLink WebSocket at {uri}")
        except Exception as e:
            print(f"Failed to connect to LiveLink WebSocket: {e}")
            self.is_connected = False
            
    async def disconnect_websocket(self):
        """Déconnecte du serveur WebSocket"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            
    async def send_blendshapes(self, blendshapes: List[float]):
        """
        Envoie les blendshapes via WebSocket ou UDP
        
        Args:
            blendshapes: Liste de 68 valeurs float
        """
        if len(blendshapes) != 68:
            raise ValueError(f"Expected 68 blendshapes, got {len(blendshapes)}")
            
        # Créer le message LiveLink
        message = self._create_livelink_message(blendshapes)
        
        # Essayer WebSocket en premier
        if self.is_connected and self.websocket:
            try:
                await self.websocket.send(json.dumps(message))
                return
            except Exception as e:
                print(f"WebSocket send failed: {e}")
                self.is_connected = False
                
        # Fallback sur UDP
        self.send_blendshapes_udp(blendshapes)
        
    def send_blendshapes_udp(self, blendshapes: List[float]):
        """
        Envoie les blendshapes via UDP (mode direct)
        
        Args:
            blendshapes: Liste de 68 valeurs float
        """
        # Formatter le paquet UDP
        packet = self._create_udp_packet(blendshapes)
        
        try:
            self.udp_socket.sendto(packet, (self.host, self.port))
        except Exception as e:
            print(f"UDP send failed: {e}")
            
    def _create_livelink_message(self, blendshapes: List[float]) -> Dict:
        """Crée un message LiveLink formaté"""
        return {
            "subject_name": "GalaFace",
            "timestamp": time.time(),
            "blendshapes": {
                name: value 
                for name, value in zip(self.blendshape_names, blendshapes)
            }
        }
        
    def _create_udp_packet(self, blendshapes: List[float]) -> bytes:
        """
        Crée un paquet UDP pour LiveLink
        
        Format: [header][timestamp][subject_len][subject][count][values]
        """
        # Header magic number
        header = b"LIVE"
        
        # Timestamp
        timestamp = struct.pack('d', time.time())
        
        # Subject name
        subject = b"GalaFace"
        subject_len = struct.pack('I', len(subject))
        
        # Blendshapes count
        count = struct.pack('I', len(blendshapes))
        
        # Blendshapes values
        values = struct.pack(f'{len(blendshapes)}f', *blendshapes)
        
        # Assembler le paquet
        packet = header + timestamp + subject_len + subject + count + values
        
        return packet
        
    def close(self):
        """Ferme les connexions"""
        if self.udp_socket:
            self.udp_socket.close()
            
    async def aclose(self):
        """Ferme les connexions async"""
        await self.disconnect_websocket()
        self.close()