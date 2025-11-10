#!/usr/bin/env python3
"""
Script de création de la structure complète du projet Agent IA Frigorifique
"""

import os
import shutil
import sys

def creer_structure_projet():
    print("🚀 Création de la structure du projet Agent IA Frigorifique...")
    
    # Dossier racine
    dossier_racine = "agent_ia_frigo"
    
    # Créer le dossier racine
    if os.path.exists(dossier_racine):
        print(f"⚠️  Le dossier '{dossier_racine}' existe déjà.")
        reponse = input("Voulez-vous le supprimer et recréer ? (o/N): ")
        if reponse.lower() != 'o':
            print("❌ Arrêt du script.")
            return
        shutil.rmtree(dossier_racine)
    
    os.makedirs(dossier_racine)
    print(f"✅ Dossier racine '{dossier_racine}' créé")
    
    # Changer vers le dossier du projet
    os.chdir(dossier_racine)
    
    # Créer les sous-dossiers
    dossiers = ['models', 'datasets', 'logs']
    for dossier in dossiers:
        os.makedirs(dossier, exist_ok=True)
        print(f"✅ Dossier '{dossier}' créé")
    
    # Créer les fichiers .gitkeep
    with open('datasets/.gitkeep', 'w') as f:
        pass
    with open('logs/.gitkeep', 'w') as f:
        pass
    print("✅ Fichiers .gitkeep créés")
    
    # Créer le fichier .gitignore
    gitignore_content = """# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt

# Environnements virtuels
.env
.venv

# Logs
*.log
logs/*.log
!logs/.gitkeep

# Données temporaires
datasets/*.csv
!datasets/.gitkeep
*.csv
*.jsonl

# Système
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
ehthumbs.db
[Tt]humbs.db
"""
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    print("✅ Fichier .gitignore créé")
    
    # Créer le fichier requirements.txt
    requirements_content = """flask==3.0.0
joblib==1.3.2
pandas==2.1.4
scikit-learn==1.3.2
numpy==1.26.2
gunicorn==21.2.0
"""
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements_content)
    print("✅ Fichier requirements.txt créé")
    
    # Créer le Dockerfile
    dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements d'abord pour mieux utiliser le cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p models datasets logs

# Exposer le port
EXPOSE 5000

# Variables d'environnement
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Lancer l'application avec Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "agent_ia:app"]
"""
    with open('Dockerfile', 'w', encoding='utf-8') as f:
        f.write(dockerfile_content)
    print("✅ Dockerfile créé")
    
    # Créer le fichier train_initial_models.py
    train_models_content = '''import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

print("🎯 Génération des modèles initiaux pour le système frigorifique...")

# Créer le dossier models
os.makedirs('models', exist_ok=True)

# Définition des pannes et leurs caractéristiques
pannes_config = {
    'surchauffe_compresseur': {
        'Température': (40, 80),      # Température très élevée
        'Courant': (12, 20),          # Courant très élevé
        'Vibration': (7, 10)          # Vibration forte
    },
    'fuite_fluide': {
        'Pression_BP': (0.5, 1.5),    # Pression BP basse
        'Pression_HP': (5, 8),        # Pression HP basse
        'Température': (-5, 5)        # Température anormale
    },
    'givrage_evaporateur': {
        'Température': (-25, -15),    # Température très basse
        'Débit_air': (10, 50),        # Débit d'air faible
        'Humidité': (70, 90)          # Humidité élevée
    },
    'panne_electrique': {
        'Tension': (100, 180),        # Tension basse
        'Courant': (0, 2),            # Courant faible ou nul
        'Vibration': (0, 1)           # Vibration nulle
    },
    'obstruction_conduit': {
        'Débit_air': (10, 40),        # Débit d'air très faible
        'Pression_HP': (16, 25),      # Pression HP élevée
        'Courant': (8, 12)            # Courant élevé
    },
    'defaillance_ventilateur': {
        'Débit_air': (0, 30),         # Débit d'air très faible
        'Température': (5, 15),       # Température élevée
        'Vibration': (8, 10)          # Vibration forte
    },
    'capteur_defectueux': {
        'Température': (-100, 100),   # Valeurs extrêmes
        'Pression_BP': (-10, 10),     # Valeurs impossibles
        'Courant': (-5, 50)           # Valeurs aberrantes
    },
    'pression_anormale_HP': {
        'Pression_HP': (20, 30),      # Pression HP très élevée
        'Courant': (10, 15),          # Courant élevé
        'Température': (30, 50)       # Température élevée
    },
    'pression_anormale_BP': {
        'Pression_BP': (5, 8),        # Pression BP très élevée
        'Pression_HP': (18, 25),      # Pression HP élevée
        'Courant': (9, 13)            # Courant élevé
    },
    'defaut_degivrage': {
        'Température': (-20, -10),    # Température basse
        'Humidité': (75, 95),         # Humidité élevée
        'Débit_air': (30, 70)         # Débit d'air réduit
    },
    'defaillance_thermostat': {
        'Température': (-30, 30),     # Température incohérente
        'Courant': (2, 15),           # Courant variable
        'Pression_BP': (1, 6)         # Pression variable
    },
    'defaillance_compresseur': {
        'Courant': (0, 1),            # Courant nul ou faible
        'Pression_BP': (1, 2),        # Pressions basses
        'Pression_HP': (5, 8),        # Pressions basses
        'Vibration': (0, 1)           # Pas de vibration
    }
}

def generer_donnees_panne(panne_name, config, n_samples=300):
    """Génère des données pour une panne spécifique"""
    data = []
    
    for i in range(n_samples):
        ligne = {}
        
        if i < n_samples // 2:  # Cas de panne (50%)
            # Générer des valeurs dans les plages de panne
            for variable, (min_val, max_val) in config.items():
                ligne[variable] = np.random.uniform(min_val, max_val)
            
            # Remplir les autres variables avec des valeurs normales
            toutes_variables = ['Température', 'Pression_BP', 'Pression_HP', 'Courant', 
                              'Tension', 'Humidité', 'Débit_air', 'Vibration']
            
            for var in toutes_variables:
                if var not in ligne:
                    if var == 'Température': ligne[var] = np.random.uniform(-20, 5)
                    elif var == 'Pression_BP': ligne[var] = np.random.uniform(1.5, 3)
                    elif var == 'Pression_HP': ligne[var] = np.random.uniform(9, 14)
                    elif var == 'Courant': ligne[var] = np.random.uniform(4, 8)
                    elif var == 'Tension': ligne[var] = np.random.uniform(210, 230)
                    elif var == 'Humidité': ligne[var] = np.random.uniform(40, 70)
                    elif var == 'Débit_air': ligne[var] = np.random.uniform(100, 180)
                    elif var == 'Vibration': ligne[var] = np.random.uniform(1, 4)
            
            ligne['label'] = 1  # Panne détectée
            
        else:  # Cas normal (50%)
            ligne['Température'] = np.random.uniform(-20, 5)
            ligne['Pression_BP'] = np.random.uniform(1.5, 3)
            ligne['Pression_HP'] = np.random.uniform(9, 14)
            ligne['Courant'] = np.random.uniform(4, 8)
            ligne['Tension'] = np.random.uniform(210, 230)
            ligne['Humidité'] = np.random.uniform(40, 70)
            ligne['Débit_air'] = np.random.uniform(100, 180)
            ligne['Vibration'] = np.random.uniform(1, 4)
            ligne['label'] = 0  # Pas de panne
        
        data.append(ligne)
    
    return pd.DataFrame(data)

# Entraîner un modèle pour chaque panne
for panne_name, config in pannes_config.items():
    print(f"🧪 Entraînement du modèle: {panne_name}...")
    
    # Générer les données
    df = generer_donnees_panne(panne_name, config)
    
    # Préparer features et target
    features = ['Température', 'Pression_BP', 'Pression_HP', 'Courant', 
                'Tension', 'Humidité', 'Débit_air', 'Vibration']
    X = df[features]
    y = df['label']
    
    # Séparer train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraîner le modèle
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Évaluer
    accuracy = model.score(X_test, y_test)
    
    # Sauvegarder
    joblib.dump(model, f'models/{panne_name}.pkl')
    print(f"✅ {panne_name}: {accuracy:.2%} de précision")

print("\\n🎉 Tous les modèles ont été générés avec succès!")
print("📁 Dossier models/ créé avec 12 fichiers .pkl")
'''
    with open('train_initial_models.py', 'w', encoding='utf-8') as f:
        f.write(train_models_content)
    print("✅ Fichier train_initial_models.py créé")
    
    # Créer le fichier agent_ia.py (version simplifiée pour commencer)
    agent_ia_content = '''from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import json

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class AgentIAFrigorifique:
    def __init__(self):
        self.models = {}
        self.model_metrics = {}
        self.charge_modeles()
    
    def charge_modeles(self):
        """Charge tous les modèles de pannes depuis le dossier models/"""
        model_files = {
            'surchauffe_compresseur': 'models/surchauffe_compresseur.pkl',
            'fuite_fluide': 'models/fuite_fluide.pkl',
            'givrage_evaporateur': 'models/givrage_evaporateur.pkl',
            'panne_electrique': 'models/panne_electrique.pkl',
            'obstruction_conduit': 'models/obstruction_conduit.pkl',
            'defaillance_ventilateur': 'models/defaillance_ventilateur.pkl',
            'capteur_defectueux': 'models/capteur_defectueux.pkl',
            'pression_anormale_HP': 'models/pression_anormale_HP.pkl',
            'pression_anormale_BP': 'models/pression_anormale_BP.pkl',
            'defaut_degivrage': 'models/defaut_degivrage.pkl',
            'defaillance_thermostat': 'models/defaillance_thermostat.pkl',
            'defaillance_compresseur': 'models/defaillance_compresseur.pkl'
        }
        
        for panne, chemin in model_files.items():
            try:
                if os.path.exists(chemin):
                    self.models[panne] = joblib.load(chemin)
                    logger.info(f"✅ Modèle chargé: {panne}")
                else:
                    logger.warning(f"⚠️ Fichier manquant: {chemin}")
            except Exception as e:
                logger.error(f"❌ Erreur chargement {panne}: {e}")
    
    def extraire_features(self, donnees):
        """Extrait les features pour la prédiction"""
        return np.array([[
            donnees.get('Température', 0),
            donnees.get('Pression_BP', 0),
            donnees.get('Pression_HP', 0),
            donnees.get('Courant', 0),
            donnees.get('Tension', 0),
            donnees.get('Humidité', 0),
            donnees.get('Débit_air', 0),
            donnees.get('Vibration', 0)
        ]])
    
    def predire_panne(self, donnees):
        """Effectue la prédiction pour toutes les pannes"""
        features = self.extraire_features(donnees)
        predictions = {}
        scores = {}
        
        for panne, modele in self.models.items():
            try:
                prediction = modele.predict(features)[0]
                score = modele.predict_proba(features)[0][1]  # Probabilité classe positive
                predictions[panne] = prediction
                scores[panne] = float(score)
            except Exception as e:
                logger.error(f"Erreur prédiction {panne}: {e}")
                predictions[panne] = 0
                scores[panne] = 0.0
        
        # Trouver la panne avec le score le plus élevé
        panne_detectee = None
        score_max = 0.0
        variable_dominante = "Aucune"
        
        for panne, score in scores.items():
            if score > 0.7 and score > score_max:  # Seuil de confiance
                panne_detectee = panne
                score_max = score
                # Déterminer la variable dominante basée sur les features
                idx_max = np.argmax(features[0])
                variables = ['Température', 'Pression_BP', 'Pression_HP', 'Courant', 
                           'Tension', 'Humidité', 'Débit_air', 'Vibration']
                variable_dominante = variables[idx_max]
        
        return {
            'panne_detectee': panne_detectee,
            'score': round(score_max * 100, 2) if panne_detectee else 0.0,
            'variable_dominante': variable_dominante,
            'predictions_detail': predictions,
            'scores_detail': scores,
            'diagnostic_complet': f"Panne: {panne_detectee}, Confiance: {score_max:.2%}" if panne_detectee else "Système normal"
        }

# Initialiser l'agent IA
agent_ia = AgentIAFrigorifique()

@app.route('/')
def home():
    return jsonify({
        "status": "✅ Agent IA Frigorifique Opérationnel",
        "modeles_charges": len(agent_ia.models),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/status')
def status():
    return jsonify({
        "status": "operational",
        "modeles_charges": list(agent_ia.models.keys()),
        "total_modeles": len(agent_ia.models),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint principal pour les diagnostics"""
    try:
        donnees = request.get_json()
        
        if not donnees:
            return jsonify({"error": "Aucune donnée reçue"}), 400
        
        logger.info(f"📊 Données reçues: {donnees}")
        
        # Validation des champs requis
        champs_requis = ['Température', 'Pression_BP', 'Pression_HP', 'Courant', 
                        'Tension', 'Humidité', 'Débit_air', 'Vibration']
        
        for champ in champs_requis:
            if champ not in donnees:
                return jsonify({"error": f"Champ manquant: {champ}"}), 400
        
        # Prédiction
        resultat = agent_ia.predire_panne(donnees)
        
        # Sauvegarder le log
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'donnees': donnees,
            'prediction': resultat
        }
        
        os.makedirs('logs', exist_ok=True)
        with open(f'logs/diagnostic_{datetime.now().strftime("%Y%m%d")}.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\\n')
        
        return jsonify(resultat)
        
    except Exception as e:
        logger.error(f"❌ Erreur prédiction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Endpoint pour le réentraînement des modèles"""
    try:
        data = request.get_json()
        dataset_path = data.get('dataset_path', '/tmp/dataset_apprentissage.csv')
        compteur = data.get('compteur', 0)
        
        logger.info(f"🔄 Réentraînement demandé - Dataset: {dataset_path}, Compteur: {compteur}")
        
        # Simulation du réentraînement
        metrics = {
            'dernier_retraining': datetime.now().isoformat(),
            'compteur_total': compteur,
            'dataset_utilise': dataset_path,
            'statut': 'simule_pour_tests'
        }
        
        with open('logs/retraining_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return jsonify({
            "status": "success",
            "message": "Réentraînement simulé avec succès",
            "compteur": compteur,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur réentraînement: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/train_new_fault', methods=['POST'])
def train_new_fault():
    """Endpoint pour entraîner de nouvelles pannes"""
    try:
        data = request.get_json()
        fault_signature = data.get('fault_signature')
        dataset_content = data.get('dataset_content')
        sample_count = data.get('sample_count', 0)
        
        logger.info(f"🎓 Nouvelle panne à entraîner: {fault_signature}, Échantillons: {sample_count}")
        
        # Sauvegarder le dataset
        os.makedirs('datasets', exist_ok=True)
        dataset_path = f"datasets/nouvelle_panne_{fault_signature}.csv"
        
        with open(dataset_path, 'w') as f:
            f.write(dataset_content)
        
        # Simulation de l'entraînement
        return jsonify({
            "status": "success",
            "message": f"Nouvelle panne {fault_signature} entraînée (simulation)",
            "samples": sample_count,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur entraînement nouvelle panne: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/metrics')
def metrics():
    """Endpoint pour les métriques des modèles"""
    try:
        metrics_data = {}
        if os.path.exists('logs/retraining_metrics.json'):
            with open('logs/retraining_metrics.json', 'r') as f:
                metrics_data = json.load(f)
        
        return jsonify({
            "modeles_operationnels": list(agent_ia.models.keys()),
            "total_modeles": len(agent_ia.models),
            "dernier_retraining": metrics_data.get('dernier_retraining'),
            "compteur_total": metrics_data.get('compteur_total', 0),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
'''
    with open('agent_ia.py', 'w', encoding='utf-8') as f:
        f.write(agent_ia_content)
    print("✅ Fichier agent_ia.py créé")
    
    print("\n🎉 Structure du projet créée avec succès!")
    print("\n📋 Prochaines étapes:")
    print("1. 📥 Exécutez: python train_initial_models.py")
    print("2. 🐳 Testez localement: python agent_ia.py")
    print("3. 📤 Upload sur GitHub via l'interface web")
    print("4. 🚀 Déployez sur Render")

if __name__ == "__main__":
    creer_structure_projet()