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
import threading
import time
from logging.handlers import RotatingFileHandler

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

# Créer un dossier pour les logs
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajouter un handler pour fichier
file_handler = RotatingFileHandler(
    os.path.join(log_dir, "api_codex.log"), 
    maxBytes=10*1024*1024,  # 10 Mo maximum
    backupCount=5           # Garder 5 fichiers de sauvegarde
)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Flask app
app = Flask(__name__)

# Globals
blendshape_model = None
livelink = None
idle_thread = None
idle_running = False


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
    audio_bytes = request.data
    content_length = len(audio_bytes) if audio_bytes else 0
    logger.info(f"Requête reçue : {content_length} octets")

    if not audio_bytes:
        return jsonify({"status": "error", "message": "No audio data"}), 400

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generated = generate_facial_data_from_bytes(
        audio_bytes,
        blendshape_model,
        device,
        config
    )

    logger.info(f"Type de données générées : {type(generated)}")

    blendshapes_list = [] # Pour stocker la version liste, potentiellement multi-frame
    if isinstance(generated, np.ndarray):
        blendshapes_list = generated.tolist()
        logger.info(f"Conversion ndarray → liste, forme d'origine : {generated.shape}")
    else:
        blendshapes_list = generated # Si ce n'est pas un ndarray, on espère que c'est déjà une liste de listes ou une liste simple.

    # Traitement pour envoyer les frames
    if isinstance(blendshapes_list, list) and blendshapes_list:
        if isinstance(blendshapes_list[0], list): # Cas: liste de frames (liste de listes)
            logger.info(f"Traitement de {len(blendshapes_list)} frames de blendshapes.")
            has_actual_movement = False
            for i, frame_data in enumerate(blendshapes_list):
                if len(frame_data) == 68:
                    non_zero_in_current_frame = [v for v in frame_data if abs(v) > 0.01]
                    logger.info(f"Frame {i}: {len(non_zero_in_current_frame)}/{len(frame_data)} valeurs non nulles.")
                    if non_zero_in_current_frame:
                        has_actual_movement = True
                        logger.info(f"  Frame {i} échantillon non-nul: {non_zero_in_current_frame[:5]}")

                    logger.info(f"  Envoi Frame {i} via LiveLink")
                    send_to_livelink(frame_data)
                    # Ajouter un petit délai pour permettre à Unreal de traiter,
                    # et simuler un envoi à une certaine FPS.
                    # Si NeuroSync génère des frames à 60FPS, et que nous en avons 6,
                    # ils devraient être envoyés sur 6 * (1/60) = 0.1 secondes.
                    if i < len(blendshapes_list) - 1:
                         time.sleep(1.0 / config.get('frame_rate', 60)) # Utiliser frame_rate du config
                else:
                    logger.warning(f"Frame {i} n'a pas 68 valeurs (en a {len(frame_data)}). Non envoyé.")
            if not has_actual_movement:
                logger.warning("Tous les frames générés pour cette requête étaient (proches de) zéro.")

        elif isinstance(blendshapes_list[0], (float, int)): # Cas: une seule frame de blendshapes
            logger.info("Traitement d'un seul frame de blendshapes (liste simple).")
            if len(blendshapes_list) == 68:
                non_zero_in_current_frame = [v for v in blendshapes_list if abs(v) > 0.01]
                logger.info(f"Frame unique: {len(non_zero_in_current_frame)}/{len(blendshapes_list)} valeurs non nulles.")
                if non_zero_in_current_frame:
                    logger.info(f"  Frame unique échantillon non-nul: {non_zero_in_current_frame[:5]}")
                else:
                    logger.warning("Le frame unique était (proche de) zéro.")
                send_to_livelink(blendshapes_list)
            else:
                logger.warning(f"Frame unique n'a pas 68 valeurs (en a {len(blendshapes_list)}). Non envoyé.")
        else:
            logger.warning(f"Format de blendshapes_list[0] non reconnu pour l'envoi: {type(blendshapes_list[0])}")
    else:
        logger.warning("Aucun blendshape à envoyer (blendshapes_list est vide ou None après conversion).")

    # Le retour à jsonify peut être simplifié si le client n'a pas besoin des blendshapes en retour.
    # Ou vous pouvez décider de retourner uniquement le statut ou le premier/dernier frame.
    return jsonify({"status": "processed", "num_frames_generated": len(blendshapes_list) if isinstance(blendshapes_list, list) else 0})

@app.route('/test_livelink', methods=['GET'])
def test_livelink():
    """Envoie des blendshapes de test pour vérifier la connexion LiveLink."""
    # Créer un tableau de 68 valeurs (nombre attendu par LiveLinkNeuroSync)
    test_values = [0.0] * 68
    
    # Définir quelques valeurs pour animer le visage
    test_values[11] = 0.3  # Bouche ouverte
    
    logger.info(f"Envoi de blendshapes de test: {test_values[:5]}...")
    send_to_livelink(test_values)
    
    return jsonify({
        "status": "success", 
        "message": "Blendshapes de test envoyés"
    })


@app.route('/test_animation', methods=['GET'])
def test_animation():
    """Envoie des blendshapes de test animés pour vérifier le LiveLink."""
    # Créer un tableau de 68 valeurs
    test_values = [0.0] * 68
    
    # Définir des valeurs significatives
    test_values[0] = 0.7  # EyeBlinkLeft
    test_values[1] = 0.7  # EyeBlinkRight  
    test_values[7] = 0.4  # JawOpen
    test_values[20] = 0.3  # MouthClose
    
    logger.info(f"Test animation: envoi de blendshapes animés")
    send_to_livelink(test_values)
    
    return jsonify({
        "status": "success", 
        "message": "Blendshapes de test envoyés",
        "values": test_values
    })


def idle_animation_loop():
    """Fonction exécutée dans un thread pour envoyer des blendshapes d'idle en continu."""
    global idle_running
    logger.info("Démarrage de l'animation idle")
    
    frame_count = 0
    while idle_running:
        # Créer des blendshapes neutres avec légère respiration/clignement
        idle_values = [0.0] * 68  # Exactement 68 valeurs
        
        # Simuler respiration (léger mouvement de la poitrine/épaules)
        breathing = (1 + np.sin(frame_count / 30)) / 2 * 0.1
        
        # Simuler clignement occasionnel
        blink = 0.0
        if frame_count % 150 < 5:  # Clignement toutes les ~5 secondes
            blink = 0.8
        
        # Indices corrects pour les yeux dans ARKit (68)
        idle_values[0] = blink  # EyeBlinkLeft 
        idle_values[1] = blink  # EyeBlinkRight
        
        # Log pour débogage
        logger.info(f"Idle: envoi de {len(idle_values)} blendshapes")
        
        send_to_livelink(idle_values)
        frame_count += 1
        time.sleep(1/30)  # ~30 FPS
    
    logger.info("Arrêt de l'animation idle")


@app.route('/start_idle', methods=['GET'])
def start_idle():
    """Démarre l'animation idle en continu."""
    global idle_thread, idle_running
    
    if idle_thread and idle_thread.is_alive():
        return jsonify({"status": "warning", "message": "Animation idle déjà en cours"})
    
    idle_running = True
    idle_thread = threading.Thread(target=idle_animation_loop)
    idle_thread.daemon = True
    idle_thread.start()
    
    return jsonify({"status": "success", "message": "Animation idle démarrée"})


@app.route('/stop_idle', methods=['GET'])
def stop_idle():
    """Arrête l'animation idle."""
    global idle_running
    
    idle_running = False
    return jsonify({"status": "success", "message": "Animation idle arrêtée"})


if __name__ == '__main__':
    logger.info("=== Démarrage API Codex v1 ===")
    load_neurosync_model()
    init_livelink()
    app.run(host='0.0.0.0', port=API_PORT, debug=False)
