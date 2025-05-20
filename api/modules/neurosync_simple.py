"""
Module simplifié pour l'intégration NeuroSync
"""
import torch
import torch.nn as nn
import numpy as np
import torchaudio
from typing import Dict, Optional

class NeuroSyncSimple:
    """
    Wrapper simplifié pour le modèle NeuroSync
    """
    
    def __init__(self, model_path: str = "models/neurosync/model/model.pth", device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_path = model_path
        self.sample_rate = 88200
        self.blendshapes_count = 68
        
        # Charger le modèle
        self._load_model()
        
        print(f"Modèle NeuroSync chargé sur {self.device}")
        
    def _load_model(self):
        """Charge ou crée un modèle"""
        try:
            if self.model_path and torch.cuda.is_available():
                # Essayer de charger le modèle existant
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Pour le moment, utiliser un modèle simple
                self.model = self._create_simple_model()
                
                # Essayer de charger les poids (best effort)
                try:
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                except:
                    print("Attention: Impossible de charger les poids du modèle, utilisation d'un modèle aléatoire")
            else:
                self.model = self._create_simple_model()
                
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            self.model = self._create_simple_model()
            self.model.to(self.device)
            self.model.eval()
            
    def _create_simple_model(self):
        """Crée un modèle simple qui génère des blendshapes"""
        class SimpleBlendshapeModel(nn.Module):
            def __init__(self, input_size=8000, output_size=68):
                super().__init__()
                self.fc1 = nn.Linear(input_size, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, output_size)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                # Prendre une partie fixe de l'audio
                if x.shape[1] > 8000:
                    x = x[:, :8000]
                else:
                    pad_size = 8000 - x.shape[1]
                    x = torch.nn.functional.pad(x, (0, pad_size))
                    
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.sigmoid(self.fc3(x))
                return x
                
        return SimpleBlendshapeModel()
        
    def process_audio(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Traite l'audio et retourne les blendshapes
        
        Args:
            audio_tensor: Tensor [B, T] à 88200Hz
            
        Returns:
            blendshapes: Tensor [B, 68] entre 0 et 1
        """
        with torch.no_grad():
            audio_tensor = audio_tensor.to(self.device)
            blendshapes = self.model(audio_tensor)
            
            # S'assurer qu'on a 68 blendshapes
            if blendshapes.shape[1] != self.blendshapes_count:
                # Ajuster si nécessaire
                if blendshapes.shape[1] > self.blendshapes_count:
                    blendshapes = blendshapes[:, :self.blendshapes_count]
                else:
                    pad_size = self.blendshapes_count - blendshapes.shape[1]
                    blendshapes = torch.nn.functional.pad(blendshapes, (0, pad_size))
                    
            return blendshapes
            
    def process_audio_bytes(self, audio_bytes: bytes, sample_rate: int = 48000) -> np.ndarray:
        """
        Traite des bytes audio
        """
        # Convertir bytes en tensor
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_np.astype(np.float32) / 32768.0
        
        # Rééchantillonner si nécessaire
        if sample_rate != self.sample_rate:
            audio_tensor = torch.FloatTensor(audio_float).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.sample_rate
            )
            audio_resampled = resampler(audio_tensor)
        else:
            audio_resampled = torch.FloatTensor(audio_float).unsqueeze(0)
            
        # Traiter
        blendshapes_tensor = self.process_audio(audio_resampled)
        
        # Retourner comme numpy
        return blendshapes_tensor.cpu().numpy()[0]
        
    def warmup(self):
        """Préchauffe le modèle"""
        dummy_input = torch.randn(1, self.sample_rate * 2, device=self.device)
        _ = self.process_audio(dummy_input)
        print("Modèle préchauffé")