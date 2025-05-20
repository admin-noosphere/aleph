# Service OpenAI TTS pour Gala

Ce module fournit une intégration du service de synthèse vocale (TTS) d'OpenAI avec Pipecat pour le projet Gala.

## Fonctionnalités

- Intégration complète avec le pipeline Pipecat
- Support des modèles OpenAI TTS (gpt-4o-mini-tts et tts-1)
- Choix de différentes voix (alloy, echo, fable, onyx, nova, shimmer, etc.)
- Instructions de style pour personnaliser la voix
- Streaming audio pour une réponse rapide

## Configuration

1. Assurez-vous d'avoir une clé API OpenAI valide dans votre fichier `.env` :
   ```
   OPENAI_API_KEY=sk-...
   ```

2. Pour utiliser OpenAI TTS au lieu d'ElevenLabs, configurez la variable d'environnement :
   ```
   TTS_SERVICE=openai
   ```

## Utilisation

### Dans le pipeline Pipecat

Le service est automatiquement configuré dans `main.py` si `TTS_SERVICE=openai` est défini.

### Utilisation manuelle

```python
from src.voice.openai_fm import OpenAITTSService

# Initialiser le service
tts = OpenAITTSService(
    api_key="votre_clé_api_openai",
    model="gpt-4o-mini-tts",  # ou "tts-1"
    voice="nova",             # alloy, echo, fable, onyx, nova, shimmer
    instructions="Ton: Naturel et bienveillant. Émotion: Engagement et attention."
)

# Utiliser le service
async def example():
    segment = await tts.tts("Bonjour, comment puis-je vous aider aujourd'hui?")
    # segment contient les données audio PCM
```

## Scripts d'exemple

- `openai_test.py` : Démontre l'utilisation directe de l'API OpenAI TTS avec un exemple de voix "emo teenager"
- `openai_fm.py` : Module principal avec le service TTS compatible Pipecat

## Voix disponibles

- `alloy` : Voix neutre et équilibrée
- `echo` : Voix plus grave et posée
- `fable` : Voix chaleureuse et douce
- `onyx` : Voix grave et autoritaire
- `nova` : Voix féminine et professionnelle
- `shimmer` : Voix claire et dynamique

## Instructions de style

Vous pouvez personnaliser la voix avec des instructions comme :

```
Ton: Enthousiaste et dynamique.
Émotion: Joie et optimisme.
Débit: Rapide avec des accents sur les mots clés.
```

ou

```
Ton: Calme et posé.
Émotion: Sérénité et confiance.
Débit: Lent avec des pauses réfléchies.
``` 