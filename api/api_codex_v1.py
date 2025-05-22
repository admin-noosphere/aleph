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
import io # Added import
import wave # Added import
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

# Configuration from environment variables
LIVELINK_IP = os.getenv("LIVELINK_IP", "192.168.1.14")
LIVELINK_PORT = 11111 # Kept as constant, can be made env-configurable if needed
API_PORT = int(os.getenv("API_PORT", "6969"))
AUTO_IDLE_TIMEOUT_SECONDS = int(os.getenv("AUTO_IDLE_TIMEOUT_SECONDS", "5"))

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
idle_thread = None # Will hold the thread object for the idle animation loop
monitor_thread = None # Will hold the thread object for the activity monitor
idle_running = False # Master switch for whether idle animation should be active
last_audio_time = time.time() # Timestamp of the last audio processing activity


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
        "port": API_PORT, # Loaded from env
        "livelink_ip": LIVELINK_IP, # Loaded from env
        "livelink_port": LIVELINK_PORT,
        "auto_idle_timeout": AUTO_IDLE_TIMEOUT_SECONDS # Added for info
    })


@app.route('/audio_to_blendshapes', methods=['POST'])
def audio_to_blendshapes_route():
    global idle_running, last_audio_time # Added globals
    audio_bytes = request.data
    content_length = len(audio_bytes) if audio_bytes else 0
    # logger.info(f"Requête reçue : {content_length} octets") # Original log, will be covered by WAV log or warning

    if not audio_bytes:
        logger.warning("Audio_to_blendshapes: No audio data received.")
        return jsonify({"status": "error", "message": "No audio data"}), 400

    # Log WAV characteristics
    try:
        with io.BytesIO(audio_bytes) as wav_file_like:
            with wave.open(wav_file_like, 'rb') as wf:
                actual_sr = wf.getframerate()
                actual_channels = wf.getnchannels()
                actual_swidth = wf.getsampwidth()
                num_frames = wf.getnframes()
                logger.info(f"Received audio WAV data: SR={actual_sr}, Channels={actual_channels}, SampleWidth={actual_swidth}, Frames={num_frames}, Size={len(audio_bytes)} bytes")
    except Exception as e:
        logger.warning(f"Could not parse incoming audio as WAV: {e}. Assuming raw PCM or expected format for NeuroSync. Size={len(audio_bytes)} bytes")

    # Update activity timestamp and manage idle state
    last_audio_time = time.time()
    if idle_running:
        logger.info("Audio data received: stopping active idle animation.")
        idle_running = False

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


def managed_idle_animation_loop():
    """Fonction exécutée dans un thread pour envoyer des blendshapes d'idle en continu."""
    global idle_running, livelink, logger
    logger.info("Démarrage du thread managed_idle_animation_loop.")
    
    frame_count = 0
    while True:
        if idle_running:
            # Créer des blendshapes neutres avec légère respiration/clignement
            idle_values = [0.0] * 68  # Exactement 68 valeurs
            
            # Simuler respiration (léger mouvement de la poitrine/épaules)
            # Note: This breathing simulation is very subtle.
            breathing = (1 + np.sin(frame_count / 30.0)) / 2.0 * 0.05 # Reduced intensity
            
            # Simuler clignement occasionnel
            blink = 0.0
            if (frame_count % 150) < 5:  # Clignement toutes les ~5 secondes (150 frames / 30 fps = 5s)
                blink = 0.9 # Full blink
            elif (frame_count % 150) < 10: # Keep eyes closed for a bit
                blink = 0.9
            
            idle_values[0] = blink  # EyeBlinkLeft 
            idle_values[1] = blink  # EyeBlinkRight
            
            if livelink:
                send_to_livelink(idle_values)
                logger.debug(f"Sent idle frame: {frame_count}, Blink: {blink}, Breathing: {breathing:.3f}")
            frame_count += 1
        else:
            if frame_count != 0: 
                logger.debug("Idle not running. Resetting idle frame_count.")
            frame_count = 0 # Reset frame count when idle becomes inactive
            
        time.sleep(1/30.0)  # Loop runs consistently at ~30 FPS

def audio_activity_monitor():
    """Monitors for audio inactivity and triggers idle animation."""
    global idle_running, last_audio_time, logger, AUTO_IDLE_TIMEOUT_SECONDS
    logger.info("Démarrage du thread audio_activity_monitor.")
    while True:
        time.sleep(1) # Check every second
        if not idle_running and (time.time() - last_audio_time > AUTO_IDLE_TIMEOUT_SECONDS):
            logger.info(f"No audio for {AUTO_IDLE_TIMEOUT_SECONDS}s. Triggering auto-idle.")
            idle_running = True

@app.route('/start_idle', methods=['POST']) # Changed to POST for consistency
def start_idle():
    """Démarre l'animation idle en continu."""
    global idle_running, last_audio_time, logger
    
    logger.info("Manual /start_idle called.")
    last_audio_time = time.time() # Update to prevent auto-idle from immediately conflicting
    idle_running = True
    return jsonify({"status": "success", "message": "Idle animation explicitly started."})


@app.route('/stop_idle', methods=['POST']) # Changed to POST
def stop_idle():
    """Arrête l'animation idle."""
    global idle_running, logger
    
    logger.info("Manual /stop_idle called.")
    idle_running = False
    return jsonify({"status": "success", "message": "Idle animation explicitly stopped."})


if __name__ == '__main__':
 
    logger.info("=== Démarrage API Codex v1 ===")
    load_neurosync_model()
    init_livelink()

    logger.info("Starting background managed idle animation thread.")
    idle_thread = threading.Thread(target=managed_idle_animation_loop)
    idle_thread.daemon = True
    idle_thread.start()

    logger.info("Starting background audio activity monitor thread.")
    monitor_thread = threading.Thread(target=audio_activity_monitor)
    monitor_thread.daemon = True
    monitor_thread.start()

    app.run(host='0.0.0.0', port=API_PORT, debug=False)
