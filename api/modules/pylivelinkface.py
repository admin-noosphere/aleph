#!/usr/bin/env python3
"""
Module PyLiveLinkFace pour Gala v1
Basé sur l'implémentation de NeuroSync_Player
"""

from __future__ import annotations
import datetime
import struct
import uuid
from enum import IntEnum
from typing import Optional


class FaceBlendShape(IntEnum):
    """ARKit FaceBlendShape indices (0-60)"""
    EyeBlinkLeft = 0
    EyeLookDownLeft = 1
    EyeLookInLeft = 2
    EyeLookOutLeft = 3
    EyeLookUpLeft = 4
    EyeSquintLeft = 5
    EyeWideLeft = 6
    EyeBlinkRight = 7
    EyeLookDownRight = 8
    EyeLookInRight = 9
    EyeLookOutRight = 10
    EyeLookUpRight = 11
    EyeSquintRight = 12
    EyeWideRight = 13
    JawForward = 14
    JawLeft = 15
    JawRight = 16
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
    MouthShrugLower = 33
    MouthShrugUpper = 34
    MouthPressLeft = 35
    MouthPressRight = 36
    MouthLowerDownLeft = 37
    MouthLowerDownRight = 38
    MouthUpperUpLeft = 39
    MouthUpperUpRight = 40
    BrowDownLeft = 41
    BrowDownRight = 42
    BrowInnerUp = 43
    BrowOuterUpLeft = 44
    BrowOuterUpRight = 45
    CheekPuff = 46
    CheekSquintLeft = 47
    CheekSquintRight = 48
    NoseSneerLeft = 49
    NoseSneerRight = 50
    TongueOut = 51
    # HeadYaw = 52  # Rotation horizontale tête
    # HeadPitch = 53  # Inclinaison verticale tête
    # HeadRoll = 54  # Roulis tête
    # LeftEyeYaw = 55  # Rotation horizontale oeil gauche
    # LeftEyePitch = 56  # Inclinaison verticale oeil gauche 
    # LeftEyeRoll = 57  # Roulis oeil gauche
    # RightEyeYaw = 58  # Rotation horizontale oeil droit
    # RightEyePitch = 59  # Inclinaison verticale oeil droit
    # RightEyeRoll = 60  # Roulis oeil droit


class PyLiveLinkFace:
    """
    Implémente le protocole LiveLink pour les animations faciales
    Compatible avec Unreal Engine 5's LiveLink Faces
    """
    
    def __init__(self, name: str = "GalaFace", uuid_str: str = None, fps: int = 60) -> None:
        # Générer un UUID si non fourni
        if uuid_str is None:
            uuid_str = str(uuid.uuid1())
        # Ajouter le préfixe $ comme NeuroSync_Player
        self.uuid = f"${uuid_str}" if not uuid_str.startswith("$") else uuid_str
        self.name = name
        self.fps = fps
        self._version = 6  # Version du protocole LiveLink
        
        # Timestamp et frame info
        self._frames = 0
        self._sub_frame = 0
        self._denominator = 1
        
        # 61 blendshapes (52 ARKit + 9 custom)
        self._blend_shapes = [0.0] * 61
        
        # Facteurs d'échelle (optionnels)
        self._scaling_factor_mouth = 1.0
        self._scaling_factor_eyes = 1.0
        self._scaling_factor_eyebrows = 1.0
        
        # Initialiser le timestamp
        self._update_timestamp()
    
    def _update_timestamp(self):
        """Met à jour le timestamp et les infos de frame"""
        now = datetime.datetime.now()
        total_seconds = (now.hour * 3600 + now.minute * 60 + 
                        now.second + now.microsecond / 1000000.0)
        self._frames = int(total_seconds * self.fps)
        self._sub_frame = int((total_seconds * self.fps - self._frames) * 4294967296)
        self._denominator = 1
    
    def encode(self) -> bytes:
        """
        Encode les données LiveLink dans le format binaire attendu par Unreal
        
        Format binaire:
        - Version (4 bytes, UInt32)
        - UUID (36 bytes, UTF-8)
        - Name length (4 bytes, Int32 big-endian)
        - Name (variable, UTF-8)
        - Frame time (8 bytes: 4 bytes frames + 4 bytes sub_frame)
        - Frame rate (8 bytes: 4 bytes fps + 4 bytes denominator)
        - Blend shape count (1 byte, UInt8)
        - Blend shape values (61 * 4 bytes, float32 big-endian)
        """
        # Version
        version_packed = struct.pack('<I', self._version)
        
        # UUID
        uuid_packed = self.uuid.encode('utf-8')
        
        # Subject name
        name_bytes = self.name.encode('utf-8')
        name_length_packed = struct.pack('!i', len(name_bytes))
        
        # Update timestamp
        self._update_timestamp()
        
        # Frame time
        frames_packed = struct.pack("!II", self._frames, self._sub_frame)
        
        # Frame rate
        frame_rate_packed = struct.pack("!II", self.fps, self._denominator)
        
        # Blend shapes
        blend_shape_count = 61
        scaled_shapes = self._apply_scaling()
        data_packed = struct.pack('!B', blend_shape_count)
        data_packed += struct.pack('!61f', *scaled_shapes)
        
        # Combine all parts
        return (version_packed + uuid_packed + name_length_packed + 
                name_bytes + frames_packed + frame_rate_packed + data_packed)
    
    def _apply_scaling(self) -> list:
        """Applique les facteurs d'échelle aux blendshapes"""
        scaled = self._blend_shapes.copy()
        
        # Échelle pour la bouche (indices 18-40)
        for i in range(18, 41):
            scaled[i] *= self._scaling_factor_mouth
        
        # Échelle pour les yeux (indices 0-13)
        for i in range(0, 14):
            scaled[i] *= self._scaling_factor_eyes
        
        # Échelle pour les sourcils (indices 41-45)
        for i in range(41, 46):
            scaled[i] *= self._scaling_factor_eyebrows
        
        return scaled
    
    def set_blendshape(self, index: int, value: float) -> None:
        """
        Définit la valeur d'un blendshape
        
        Args:
            index: Index du blendshape (0-60)
            value: Valeur entre 0.0 et 1.0
        """
        if 0 <= index < 61:
            # Limiter les valeurs entre 0 et 1
            self._blend_shapes[index] = max(0.0, min(1.0, value))
    
    def set_blendshapes(self, values: list) -> None:
        """
        Définit toutes les valeurs des blendshapes
        
        Args:
            values: Liste de 61 valeurs float
        """
        if len(values) >= 61:
            for i in range(61):
                self.set_blendshape(i, values[i])
    
    def get_blendshape(self, index: int) -> float:
        """Récupère la valeur d'un blendshape"""
        if 0 <= index < 61:
            return self._blend_shapes[index]
        return 0.0
    
    def get_blendshapes(self) -> list:
        """Récupère toutes les valeurs des blendshapes"""
        return self._blend_shapes.copy()
    
    def reset(self):
        """Réinitialise tous les blendshapes à 0"""
        self._blend_shapes = [0.0] * 61
    
    def encode(self) -> bytes:
        """
        Encode les données pour l'envoi LiveLink
        Compatible avec le protocole Unreal Engine LiveLink
        """
        version_packed = struct.pack('<I', self._version)
        uuid_packed = self.uuid.encode('utf-8')
        name_packed = self.name.encode('utf-8')
        name_length_packed = struct.pack('!i', len(self.name))
        
        now = datetime.datetime.now()
        # Calcul simple du timecode
        self._frames = int(now.hour * 3600 * self.fps + 
                           now.minute * 60 * self.fps + 
                           now.second * self.fps + 
                           (now.microsecond / 1000000.0) * self.fps)
        
        frames_packed = struct.pack("!II", self._frames, self._sub_frame)
        frame_rate_packed = struct.pack("!II", self.fps, self._denominator)
        
        # Empaqueter les 61 blendshapes
        data_packed = struct.pack('!B61f', 61, *self._blend_shapes)
        
        return version_packed + uuid_packed + name_length_packed + name_packed + frames_packed + frame_rate_packed + data_packed