#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Service OpenAI TTS pour Pipecat.
Compatible avec modèles OpenAI GPT-4o-mini-tts et tts-1.
"""

from __future__ import annotations

import asyncio
import io
import wave
from typing import Any, Dict, List, Optional, AsyncGenerator

from openai import AsyncOpenAI
from pipecat.pipeline.pipeline import Frame
from pipecat.services.tts_service import TTSService


# Définition d'une classe AudioSegment simple pour remplacer celle de pipecat.audio.segmenter
class AudioSegment:
    def __init__(
        self, audio: bytes, sample_rate: int, channels: int, is_end: bool = True
    ):
        self.audio = audio
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_end = is_end


class OpenAITTSService(TTSService):
    """Service TTS utilisant l'API OpenAI."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini-tts",
        voice: str = "alloy",
        instructions: Optional[str] = None,
        speech_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialise le service TTS OpenAI.

        Args:
            api_key: Clé API OpenAI
            model: Modèle TTS OpenAI (gpt-4o-mini-tts ou tts-1)
            voice: Voix à utiliser (alloy, echo, fable, onyx, nova, shimmer, etc.)
            instructions: Instructions optionnelles pour guider le style de parole
            speech_params: Paramètres supplémentaires pour l'API
        """
        TTSService.__init__(self)

        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._instructions = instructions
        self._speech_params = speech_params or {}
        self._client = AsyncOpenAI(api_key=api_key)

    async def tts(self, text: str) -> Optional[AudioSegment]:
        """
        Convertit du texte en parole via l'API OpenAI.

        Args:
            text: Texte à convertir en parole

        Returns:
            Segment audio ou None en cas d'erreur
        """
        if not text.strip():
            return None

        # Paramètres pour l'API
        params = {
            "model": self._model,
            "voice": self._voice,
            "input": text,
            "response_format": "pcm",
        }

        # Ajouter les instructions si présentes
        if self._instructions:
            params["instructions"] = self._instructions

        # Ajouter les paramètres supplémentaires
        params.update(self._speech_params)

        try:
            # Appel à l'API OpenAI avec streaming
            audio_bytes = bytearray()

            async with self._client.audio.speech.with_streaming_response.create(
                **params
            ) as response:
                async for chunk in response.iter_bytes():
                    audio_bytes.extend(chunk)

            # Format PCM est déjà 24000 Hz, 16-bit, mono
            sample_rate = 24000
            channels = 1

            # Créer un segment audio
            return AudioSegment(
                audio=bytes(audio_bytes),
                sample_rate=sample_rate,
                channels=channels,
                is_end=True,
            )

        except Exception as e:
            self.logger.error(f"Erreur lors de la synthèse vocale OpenAI: {e}")
            return None

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """
        Méthode abstraite requise par TTSService.
        Convertit du texte en parole et renvoie des frames audio.

        Args:
            text: Texte à convertir en parole

        Yields:
            Frames audio
        """
        if not text.strip():
            return

        segment = await self.tts(text)
        if segment:
            yield Frame(
                source=self.name,
                text=text,
                audio=segment,
                meta={"is_end": True},
            )

    async def process_frame(self, frame: Frame, direction) -> List[Frame]:
        """Convertit le texte de la frame en audio."""
        # Vérifier si c'est une StartFrame ou un autre type de frame spécial
        if not hasattr(frame, "text"):
            # Simplement passer la frame telle quelle
            return [frame]

        if not frame.text:
            return [frame]

        segment = await self.tts(frame.text)
        if not segment:
            return [frame]

        return [
            Frame(
                source=self.name,
                text=frame.text,
                audio=segment,
                meta=frame.meta,
            )
        ]


def pcm_to_wav_bytes(pcm: bytes, sr: int, channels: int = 1) -> bytes:
    """Convertit des données PCM en format WAV."""
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm)
        return buf.getvalue()


# Test simple si exécuté directement
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    async def test_tts():
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY non définie dans les variables d'environnement")
            return

        tts = OpenAITTSService(
            api_key=api_key,
            model="gpt-4o-mini-tts",
            voice="nova",
            instructions="Ton: Naturel et bienveillant. Émotion: Engagement et attention.",
        )

        text = "Bonjour, je suis un assistant virtuel. Comment puis-je vous aider aujourd'hui?"
        segment = await tts.tts(text)

        if segment:
            # Enregistrer en WAV pour test
            wav_data = pcm_to_wav_bytes(
                segment.audio, segment.sample_rate, segment.channels
            )
            with open("test_tts.wav", "wb") as f:
                f.write(wav_data)
            print("Fichier test_tts.wav créé avec succès.")
        else:
            print("Échec de la synthèse vocale.")

    asyncio.run(test_tts())
