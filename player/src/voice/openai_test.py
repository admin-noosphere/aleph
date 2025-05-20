#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test direct du service OpenAI TTS avec les mêmes paramètres que dans main.py
"""
import asyncio
import os
import sys
import wave
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Ajout du chemin de Pipecat local au PYTHONPATH
vendor_path = Path(__file__).resolve().parent.parent.parent / "vendor"
sys.path.insert(0, str(vendor_path / "pipecat" / "src"))

# Chargement des variables d'environnement
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("La variable OPENAI_API_KEY est manquante dans .env")

openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Texte de test - réponse typique que le LLM pourrait générer
input_text = """Bonjour! Comment puis-je vous aider aujourd'hui? Je suis là pour répondre à vos questions ou discuter de ce qui vous intéresse."""


# Fonction de lecture audio
async def play_audio_with_pyaudio(
    audio_data, sample_rate=24000, channels=1, sample_width=2
):
    try:
        import pyaudio
        import numpy as np

        p = pyaudio.PyAudio()

        # Convertir les données PCM en tableau numpy
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Ouvrir un flux audio
        stream = p.open(
            format=p.get_format_from_width(sample_width),
            channels=channels,
            rate=sample_rate,
            output=True,
        )

        # Jouer l'audio
        stream.write(audio_array.tobytes())

        # Nettoyer
        stream.stop_stream()
        stream.close()
        p.terminate()

        print("Audio joué avec succès")

    except ImportError:
        print("pyaudio n'est pas installé. Installation avec: pip install pyaudio")
        return
    except Exception as e:
        print(f"Erreur lors de la lecture audio: {e}")


async def main() -> None:
    print("=== Test du service OpenAI TTS ===")
    print("Modèle: tts-1, Voix: nova")
    print("Texte:", input_text)
    print("\nGénération de l'audio...")

    try:
        # Génération audio sans le paramètre instructions
        response = await openai.audio.speech.create(
            model="tts-1",  # Utiliser tts-1 au lieu de gpt-4o-mini-tts
            voice="nova",
            input=input_text,
            response_format="pcm",
        )

        # Récupérer les données audio
        audio_data = response.content

        # Lecture audio
        print("Lecture de l'audio...")
        await play_audio_with_pyaudio(audio_data)

        # Sauvegarde en WAV
        print("Enregistrement de l'audio en fichier WAV...")
        with wave.open("test_tts.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(audio_data)

        print("Audio enregistré dans 'test_tts.wav'")
        print("Test terminé avec succès!")

    except Exception as e:
        print(f"ERREUR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

        # Afficher les modèles disponibles
        print("\nVoici les modèles disponibles dans votre version de l'API:")
        try:
            import inspect

            print(inspect.signature(openai.audio.speech.create))
        except Exception:
            print("Impossible d'afficher la signature de la méthode")


if __name__ == "__main__":
    asyncio.run(main())
