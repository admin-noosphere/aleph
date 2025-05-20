   # Aleph

   ## Installation
   Ce projet contient deux composants avec des environnements séparés :

   ### API (backend)
   ```bash
   # Avec conda
   cd api
   conda env create -f environment.yml
   conda activate aleph-api
   ```

   ### Player (client)
   ```bash
   # Avec pip
   cd player
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   ```
