import asyncio
import time
import numpy as np
from livelink import LiveLinkDataTrack, BlendshapeFrame, FaceBlendShape  # Ajouter FaceBlendShape à l'import


async def main():
    # Créer une instance LiveLinkDataTrack avec l'adresse IP correcte
    track = LiveLinkDataTrack(use_udp=True, udp_ip="192.168.1.14")
    print("LiveLink initialisé et connecté à 192.168.1.14:11111")
    
    print("Test de mouvement exagéré - Ouverture complète de la mâchoire pendant 2 secondes")
    # D'abord, une animation très visible pour vérifier que quelque chose arrive dans Unreal
    for _ in range(60):  # ~2 secondes
        values = [0.0] * 61
        values[17] = 1.0  # JawOpen à 100%
        frame = BlendshapeFrame(values, frame_idx=0)
        await track.send_blendshape(frame)
        await asyncio.sleep(0.033)
    
    # Test avec plusieurs blendshapes qui varient pour simuler une animation
    print("Animation normale commencée...")
    for i in range(300):  # Animation plus longue pour avoir le temps d'observer
        # Générer des valeurs ondulantes avec plage de mouvement augmentée
        values = [0.0] * 61
        
        # Animation avec amplitude augmentée
        values[17] = 0.7 * np.sin(i * 0.1) + 0.7  # JawOpen - plus ample
        values[19] = 0.5 * np.sin(i * 0.15) + 0.5  # MouthFunnel - plus ample
        values[20] = 0.6 * np.cos(i * 0.2) + 0.6   # MouthPucker - plus ample
        
        # Ajout de mouvements latéraux de la mâchoire
        jaw_lr = 0.3 * np.sin(i * 0.05)
        values[15] = max(0, jaw_lr)      # JawLeft
        values[16] = max(0, -jaw_lr)     # JawRight
        
        # Clignotement plus fréquent
        if i % 15 == 0:  # Toutes les 0.5 secondes environ
            values[0] = 1.0  # EyeBlinkLeft
            values[7] = 1.0  # EyeBlinkRight
        
        # Mouvement des sourcils
        values[41] = 0.4 * np.sin(i * 0.07) + 0.4  # BrowDownLeft
        values[42] = 0.4 * np.sin(i * 0.07) + 0.4  # BrowDownRight
        
        frame = BlendshapeFrame(values, frame_idx=i)
        await track.send_blendshape(frame)
        
        # Log plus détaillé pour débogage
        if i % 10 == 0:  # Log toutes les 10 frames
            print(f"Frame {i}: JawOpen={values[17]:.2f}, MouthFunnel={values[19]:.2f}, Eye Blink={values[0]:.2f}")
        
        await asyncio.sleep(0.033)  # ~30 FPS

    print("Test terminé avec succès!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Test terminé par l'utilisateur")
