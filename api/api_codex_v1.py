#!/usr/bin/env python3
"""
API Codex v1 - Pipeline PCM -> Blendshapes -> LiveLink
Cette version utilise la fonction officielle NeuroSync pour obtenir les blendshapes
et envoie les valeurs converties à Unreal Engine via LiveLinkNeuroSync.
"""

import os
import sys
import logging
import warnings
from typing import List

from flask import Flask, request, jsonify

warnings.filterwarnings("ignore")

# Configuration GPU (même approche que les autres scripts)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np

# Chemin vers l'API NeuroSync originale
neurosync_path = "/home/gieidi-prime/Agents/NeuroSync_Local_API/neurosync_v3_all copy/NeuroSync_Real-Time_API"
sys.path.insert(0, neurosync_path)

# Imports NeuroSync
from models.neurosync.config import config
from models.neurosync.generate_face_shapes import generate_facial_data_from_bytes
from models.neurosync.model.model import load_model

# Module LiveLink style NeuroSync_Player
from modules.livelink_neurosync import LiveLinkNeuroSync

# Paramètres de connexion
LIVELINK_IP = "192.168.1.14"
LIVELINK_PORT = 11111
API_PORT = 6969

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Globals
blendshape_model = None
livelink = None


def load_neurosync_model():
    """Charge le modèle NeuroSync d'origine."""
    global blendshape_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Chargement du modèle NeuroSync sur {device}")

    model_path = os.path.join(neurosync_path, "models/neurosync/model/model.pth")
    blendshape_model = load_model(model_path, config, device)
    logger.info("Modèle NeuroSync chargé")

    return blendshape_model


def init_livelink():
    """Initialise la connexion LiveLink."""
    global livelink
    livelink = LiveLinkNeuroSync(udp_ip=LIVELINK_IP, udp_port=LIVELINK_PORT, fps=60)
    logger.info(f"Connexion LiveLink prête vers {LIVELINK_IP}:{LIVELINK_PORT}")


def send_to_livelink(blendshapes: List[float]):
    """Envoie 68 blendshapes ARKit via LiveLinkNeuroSync."""
    if not livelink:
        logger.error("LiveLink non initialisé")
        return
    try:
        livelink.send_blendshapes(blendshapes)
    except Exception as e:
        logger.error(f"Erreur d'envoi LiveLink: {e}")


@app.route('/health', methods=['GET'])
def health():
    """Endpoint de santé."""
    return jsonify({
        "status": "healthy",
        "model_loaded": blendshape_model is not None,
        "livelink_connected": livelink is not None,
        "port": API_PORT,
        "livelink_ip": LIVELINK_IP,
        "livelink_port": LIVELINK_PORT,
    })


@app.route('/audio_to_blendshapes', methods=['POST'])
def audio_to_blendshapes_route():
    """Convertit un blob PCM/WAV en blendshapes et les envoie."""
    audio_bytes = request.data

    if not audio_bytes:
        return jsonify({"status": "error", "message": "No audio data"}), 400

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Utilise la fonction officielle qui gère automatiquement le format
    generated = generate_facial_data_from_bytes(
        audio_bytes,
        blendshape_model,
        device,
        config
    )

    # Convertir en liste
    if isinstance(generated, np.ndarray):
        blendshapes = generated.tolist()
    else:
        blendshapes = generated

    # Si plusieurs frames, prendre la première pour LiveLink temps réel
    if isinstance(blendshapes, list) and blendshapes and isinstance(blendshapes[0], list):
        first_frame = blendshapes[0]
    else:
        first_frame = blendshapes

    if first_frame:
        send_to_livelink(first_frame)

    return jsonify({"blendshapes": blendshapes})


if __name__ == '__main__':
    logger.info("=== Démarrage API Codex v1 ===")
    load_neurosync_model()
    init_livelink()
    app.run(host='0.0.0.0', port=API_PORT, debug=False)
