"""Envoi asynchrone de l'audio vers le service NeuroSync.

Ce module fournit une classe `NeuroSyncClient` permettant d'envoyer les
segments audio générés par le TTS vers une API NeuroSync locale afin de
obtenir une synchronisation labiale.

L'implémentation utilise simplement une requête HTTP POST mais peut être
adaptée en fonction de l'API exacte. La méthode `send_audio` est conçue
pour être appelée depuis un processor du pipeline.
"""

from __future__ import annotations

import aiohttp
import asyncio
import traceback
from typing import Callable, List
import time
from dataclasses import dataclass
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
import os


@dataclass
class NeuroSyncClient:
    """
    Client pour l'API NeuroSync Real-Time.
    Envoie des chunks audio et reçoit des blendshapes.
    """

    def __init__(self, host="127.0.0.1", port=6969):
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/audio_to_blendshapes"
        self._session = None
        self._blendshapes_callback = None
        print(f"🔵 NeuroSyncClient initialisé - API endpoint: {self.url}")

    def set_blendshapes_callback(self, callback: Callable[[List[List[float]]], None]):
        """
        Définit le callback pour recevoir les blendshapes.
        Le callback sera appelé avec une liste de listes de valeurs de blendshapes.
        """
        self._blendshapes_callback = callback

    async def _ensure_session(self):
        """S'assure qu'une session existe"""
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def send_audio(
        self, audio_data: bytes, sample_rate: int = 16000, channels: int = 1
    ) -> bool:
        """
        Envoie des données audio à l'API NeuroSync et traite les blendshapes reçus.

        Args:
            audio_data: Données audio brutes (PCM)
            sample_rate: Taux d'échantillonnage en Hz
            channels: Nombre de canaux audio

        Returns:
            True si l'envoi a réussi, False sinon
        """
        try:
            await self._ensure_session()

            start_time = time.time()

            # Vérifier si les données sont valides
            if not audio_data or len(audio_data) < 100:
                print(
                    f"⚠️ NeuroSyncClient: données audio trop petites ({len(audio_data) if audio_data else 0} octets)"
                )
                return False

            # Log des 10 premiers octets pour débogage
            print(
                f"🔍 Envoi de {len(audio_data)} octets. Début des données: {audio_data[:10].hex()}"
            )

            # Envoyer les données audio brutes
            headers = {
                "Content-Type": "application/octet-stream",
                "Sample-Rate": str(sample_rate),
                "Channels": str(channels),
            }

            print(f"⏳ Envoi à {self.url} avec headers: {headers}")

            async with self._session.post(
                self.url, data=audio_data, headers=headers, timeout=10
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ Erreur NeuroSync API: {response.status} - {error_text}")
                    return False

                # Extraire les blendshapes de la réponse
                try:
                    result = await response.json()
                    print(
                        f"✅ Réponse reçue de l'API, taille: {len(str(result))} caractères"
                    )
                except Exception as json_err:
                    print(f"❌ Erreur parsing JSON: {json_err}")
                    text_response = await response.text()
                    print(f"Réponse brute: {text_response[:200]}...")
                    return False

                if "blendshapes" in result and isinstance(result["blendshapes"], list):
                    blendshapes = result["blendshapes"]

                    # Appeler le callback si défini
                    if self._blendshapes_callback is not None:
                        try:
                            self._blendshapes_callback(blendshapes)
                        except Exception as cb_err:
                            print(f"❌ Erreur dans le callback blendshapes: {cb_err}")
                            traceback.print_exc()

                    proc_time = time.time() - start_time
                    print(
                        f"✅ NeuroSync: reçu {len(blendshapes)} frames de blendshapes en {proc_time:.3f}s"
                    )
                    return True
                else:
                    print(
                        f"⚠️ NeuroSync: pas de blendshapes dans la réponse. Contenu: {list(result.keys())}"
                    )
                    return False

        except aiohttp.ClientError as e:
            print(f"❌ NeuroSyncClient erreur de connexion: {e}")
            return False
        except asyncio.TimeoutError:
            print("❌ NeuroSyncClient timeout")
            return False
        except Exception as e:
            print(f"❌ NeuroSyncClient erreur générale: {e}")
            traceback.print_exc()
            return False

    async def close(self):
        """Ferme la session HTTP"""
        if self._session:
            await self._session.close()
            self._session = None


class NeuroSyncProcessor(FrameProcessor):
    """Processor pipeline qui transmet chaque segment audio à NeuroSync."""

    def __init__(self, client: NeuroSyncClient):
        # nom lisible dans les logs
        super().__init__(name="neurosync_processor")
        self.client = client

    async def process_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        audio = getattr(frame, "audio", None)
        if audio and hasattr(audio, "audio"):
            await self.client.send_audio(
                audio.audio, audio.sample_rate, getattr(audio, "channels", 1)
            )
        # propage le frame inchangé
        await self.push_frame(frame, direction)
