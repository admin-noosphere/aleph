"""
Processeur pour NeuroSync - Traite les chunks audio et gère les blendshapes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import logging
import time

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


@dataclass
class VisemeFrame:
    """Frame contenant un visème"""

    id: int  # ID du visème
    time_ms: int  # Timing en millisecondes
    pts: int = 0  # Ajouter cet attribut pour compatibilité avec Daily


@dataclass
class BlendshapeFrame:
    """Frame contenant des blendshapes complets"""

    values: list[float]  # Tableau de valeurs (0-1) pour chaque blendshape
    frame_idx: int  # Index du frame
    pts: int = 0  # Ajouter cet attribut pour compatibilité avec Daily


@dataclass
class NeuroSyncBufferConfig:
    """Configuration pour le buffer NeuroSync"""

    min_frames: int = 9  # Nombre minimum de frames à accumuler avant envoi
    sample_rate: int = 16000  # Taux d'échantillonnage en Hz
    bytes_per_frame: int = 470  # Taille d'un frame audio en octets
    debug_save: bool = False  # Sauvegarder les chunks pour debug
    log_dir: Path = Path("../../logs/neurosync")  # Dossier de logs


class NeuroSyncBufferProcessor(FrameProcessor):
    """
    Processeur qui accumule l'audio et l'envoie au serveur NeuroSync,
    puis traite les blendshapes reçus.
    """

    def __init__(self, client, livelink_processor=None, config=None):
        super().__init__(name="neurosync_buffer")
        self.client = client
        self.livelink_processor = livelink_processor  # Référence au processeur LiveLink
        self.config = config or NeuroSyncBufferConfig()
        self.config.log_dir.mkdir(exist_ok=True, parents=True)
        self._logger = logging.getLogger(__name__)

        self._buffer = bytearray()
        self._viseme_queue = []
        self._blendshape_queue = []

        # Configurer le callback pour les blendshapes/visèmes
        if hasattr(self.client, "set_blendshapes_callback"):
            self.client.set_blendshapes_callback(self._handle_blendshapes)

        # Horodatage pour générer des noms de fichiers uniques
        self._chunk_counter = 0

    def _log_blendshapes_details(self, blendshapes):
        """Affiche les détails des blendshapes pour le débogage"""
        if not blendshapes or not isinstance(blendshapes, list):
            logging.info("⚠️ Blendshapes vides ou format invalide!")
            return

        # Prend un échantillon des blendshapes (jusqu'à 3 frames)
        sample_size = min(3, len(blendshapes))
        sample = blendshapes[:sample_size]
        
        for i, frame in enumerate(sample):
            if isinstance(frame, list):
                # Trouver les 5 blendshapes avec les valeurs les plus élevées
                significant_values = [(idx, val) for idx, val in enumerate(frame) if val > 0.05]
                sorted_values = sorted(significant_values, key=lambda x: x[1], reverse=True)[:5]
                
                if sorted_values:
                    values_str = ", ".join([f"idx_{idx}: {val:.2f}" for idx, val in sorted_values])
                    logging.info(f"📊 Frame {i} - Top blendshapes: {values_str} (sur {len(significant_values)} significatifs)")
                else:
                    logging.info(f"⚠️ Frame {i} - Aucune valeur significative trouvée! Max={max(frame) if frame else 0}")
            else:
                logging.info(f"⚠️ Frame {i} - Format invalide: {type(frame)}")

    def _handle_blendshapes(self, blendshapes):
        """
        Callback appelé quand des blendshapes sont reçus du serveur NeuroSync.
        Cette méthode est appelée depuis un thread du client, pas depuis le thread asyncio.
        """
        try:
            if not blendshapes or not isinstance(blendshapes, list):
                logging.info("⚠️ Reçu des blendshapes vides ou format invalide!")
                return

            # Log des détails des blendshapes reçus
            logging.info(f"📥 Reçu {len(blendshapes)} frames de blendshapes")
            self._log_blendshapes_details(blendshapes)

            # Extraire les visèmes à partir des blendshapes
            # Généralement, on utilise les valeurs de bouche pour déterminer le visème
            for i, frame in enumerate(blendshapes):
                if len(frame) < 40:  # Vérifier qu'on a assez de valeurs
                    logging.info(f"⚠️ Frame {i}: trop court ({len(frame)} valeurs < 40)")
                    continue

                # Déterminer un ID de visème à partir des valeurs de blendshapes
                # Cette logique dépend de votre modèle spécifique - à adapter
                jaw_open = frame[17] if len(frame) > 17 else 0
                mouth_funnel = frame[19] if len(frame) > 19 else 0
                mouth_pucker = frame[20] if len(frame) > 20 else 0

                # Log des valeurs clés de la bouche
                logging.info(f"👄 Frame {i}: jaw_open={jaw_open:.2f}, mouth_funnel={mouth_funnel:.2f}, mouth_pucker={mouth_pucker:.2f}")

                # Logique simplifiée pour déterminer le visème
                viseme_id = 0  # Neutre par défaut
                if jaw_open > 0.3:
                    if mouth_funnel > 0.3:
                        viseme_id = 1  # "aa"
                    elif mouth_pucker > 0.3:
                        viseme_id = 3  # "oo"
                    else:
                        viseme_id = 2  # "ah"

                # Créer des frames mais ne pas les ajouter à la file d'attente
                # Ils seront directement traités par LiveLink
                time_ms = i * 33  # ~30fps = ~33ms par frame
                viseme_frame = VisemeFrame(viseme_id, time_ms)
                blendshape_frame = BlendshapeFrame(frame, i)
                
                # Log du visème résultant
                logging.info(f"👄 Visème généré: id={viseme_id} à t={time_ms}ms")
                
                # Stocker pour envoi asynchrone via asyncio.create_task
                self._viseme_queue.append(viseme_frame)
                self._blendshape_queue.append(blendshape_frame)

            print(f"✅ Traité {len(blendshapes)} frames de blendshapes")

        except Exception as e:
            logging.error(f"❌ Erreur de traitement des blendshapes: {e}")

    async def process_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        # Important: toujours appeler super()
        await super().process_frame(frame, direction)
        
        # Traiter les files d'attente de visèmes et blendshapes
        if self.livelink_processor and (self._viseme_queue or self._blendshape_queue):
            while self._viseme_queue:
                viseme_frame = self._viseme_queue.pop(0)
                # Appel direct au lieu de pousser dans le pipeline
                if self.livelink_processor:
                    logging.info(f"📤 Envoi visème {viseme_frame.id} à LiveLink")
                    asyncio.create_task(self.livelink_processor.send_viseme(viseme_frame))

            while self._blendshape_queue:
                blendshape_frame = self._blendshape_queue.pop(0)
                # Vérification rapide des valeurs avant envoi
                if blendshape_frame.values:
                    max_value = max(blendshape_frame.values) if blendshape_frame.values else 0
                    significant_count = sum(1 for v in blendshape_frame.values if v > 0.05)
                    # Afficher les indices des 5 valeurs les plus importantes
                    if significant_count > 0:
                        top_indices = sorted([(i, v) for i, v in enumerate(blendshape_frame.values) if v > 0.05], 
                                           key=lambda x: x[1], reverse=True)[:5]
                        values_str = ", ".join([f"idx_{idx}: {val:.2f}" for idx, val in top_indices])
                        logging.info(f"📊 Envoi blendshape avec valeurs: {values_str} ({significant_count} significatives)")
                    else:
                        logging.info(f"⚠️ Blendshape: toutes les valeurs sont proches de zéro!")
                
                # Appel direct au lieu de pousser dans le pipeline
                if self.livelink_processor:
                    asyncio.create_task(self.livelink_processor.send_blendshape(blendshape_frame))

        # Traitement de l'audio uniquement en aval
        if (
            direction == FrameDirection.DOWNSTREAM
            and hasattr(frame, "audio")
            and frame.audio
        ):
            audio_data = frame.audio
            if audio_data and isinstance(audio_data, bytes):
                # Ajouter au buffer
                self._buffer.extend(audio_data)

                # Vérifier si on a assez de données pour envoyer
                min_size = self.config.min_frames * self.config.bytes_per_frame
                if len(self._buffer) >= min_size:
                    # Envoyer les données
                    chunk_size = len(self._buffer)
                    print(
                        f"⚡ NeuroSync: envoi de {chunk_size} octets (~{chunk_size/self.config.bytes_per_frame:.1f} frames)"
                    )

                    # Debug: sauvegarder le chunk
                    if self.config.debug_save:
                        self._chunk_counter += 1
                        debug_path = (
                            self.config.log_dir / f"chunk_{self._chunk_counter:03d}.pcm"
                        )
                        with open(debug_path, "wb") as f:
                            f.write(self._buffer)

                    # Envoyer de façon non-bloquante
                    asyncio.create_task(
                        self.client.send_audio(
                            bytes(self._buffer), sample_rate=self.config.sample_rate
                        )
                    )

                    # Vider le buffer
                    self._buffer.clear()

        # Propager le frame d'origine (mais pas les visèmes)
        await self.push_frame(frame, direction)
