#!/usr/bin/env python3
"""
Utilitaire pour nettoyer les processus Python qui utilisent le GPU
Exécuter ce script avant de lancer un nouveau projet pour libérer les ressources GPU
"""

import os
import subprocess
import time
import signal
import sys

def get_gpu_processes():
    """Récupère tous les PID des processus Python utilisant le GPU"""
    try:
        # Exécuter nvidia-smi pour obtenir les processus
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
            capture_output=True, text=True, check=True
        )
        
        python_pids = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    pid_str, process_name = line.strip().split(',')
                    pid = int(pid_str.strip())
                    if 'python' in process_name.strip().lower():
                        python_pids.append(pid)
                except ValueError:
                    continue
        
        return python_pids
    except Exception as e:
        print(f"Erreur lors de la récupération des processus GPU: {e}")
        return []

def kill_processes(pids, force=False):
    """Tue les processus avec les PID spécifiés"""
    my_pid = os.getpid()  # Ne pas tuer ce script
    
    for pid in pids:
        if pid != my_pid:
            try:
                # Vérifier si le processus est toujours en cours d'exécution
                os.kill(pid, 0)
                
                # Envoyer SIGTERM ou SIGKILL
                sig = signal.SIGKILL if force else signal.SIGTERM
                os.kill(pid, sig)
                print(f"Process {pid} terminé {'(forcé)' if force else ''}")
            except ProcessLookupError:
                print(f"Processus {pid} n'existe pas ou a déjà terminé")
            except PermissionError:
                print(f"Permission refusée pour terminer le processus {pid}")

def empty_gpu_cache():
    """Vide le cache CUDA si possible"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cache CUDA vidé")
        else:
            print("CUDA n'est pas disponible")
    except ImportError:
        print("PyTorch n'est pas installé, impossible de vider le cache CUDA")

def main():
    """Fonction principale"""
    print("=== Nettoyage des ressources GPU ===")
    
    # Obtenir les processus Python utilisant le GPU
    python_pids = get_gpu_processes()
    
    if not python_pids:
        print("Aucun processus Python n'utilise actuellement le GPU")
    else:
        print(f"Détecté {len(python_pids)} processus Python utilisant le GPU: {python_pids}")
        
        # Essayer de terminer proprement d'abord
        kill_processes(python_pids, force=False)
        
        # Attendre un peu
        time.sleep(2)
        
        # Vérifier quels processus sont encore en vie
        remaining_pids = [pid for pid in python_pids if pid in get_gpu_processes()]
        
        if remaining_pids:
            print(f"{len(remaining_pids)} processus persistent, application de SIGKILL...")
            kill_processes(remaining_pids, force=True)
    
    # Vider le cache CUDA
    empty_gpu_cache()
    
    print("=== Nettoyage terminé ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())