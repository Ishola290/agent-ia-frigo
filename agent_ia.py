from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

app = Flask(__name__)

# Dictionnaire des pannes et variables associées
pannes = {
    "surchauffe_compresseur": ["Température", "Courant", "Vibration"],
    "fuite_fluide": ["Pression_BP", "Température", "Courant"],
    "givrage_evaporateur": ["Température", "Humidité", "Débit_air"],
    "panne_electrique": ["Tension", "Courant"],
    "obstruction_conduit": ["Débit_air", "Pression_BP"],
    "défaillance_ventilateur": ["Débit_air", "Humidité"],
    "capteur_defectueux": ["Température", "Courant"],
    "pression_anormale_HP": ["Pression_HP", "Courant"],
    "pression_anormale_BP": ["Pression_BP", "Température"],
    "défaut_dégivrage": ["Température", "Débit_air"],
    "défaillance_thermostat": ["Température", "Courant"],
    "défaillance_compresseur": ["Courant", "Vibration"]
}

# Dossier contenant les modèles
model_dir = os.path.join(os.path.dirname(__file__), "models")
dataset_dir = os.path.join(os.path.dirname(__file__), "datasets")
log_dir = os.path.join(os.path.dirname(__file__), "logs")

# Créer les dossiers s'ils n'existent pas
for directory in [model_dir, dataset_dir, log_dir]:
    os.makedirs(directory, exist_ok=True)

def log_action(action, details):
    """Enregistre les actions dans un fichier de log"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "action": action,
        "details": details
    }
    
    log_file = os.path.join(log_dir, f"agent_ia_{datetime.now().strftime('%Y%m%d')}.log")
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction standard"""
    data = request.get_json(force=True)
    resultat = {
        "panne_detectée": None,
        "variable_dominante": None,
        "score": None,
        "diagnostic_complet": {},
        "pannes_detectees": []  # Nouveau champ pour toutes les pannes détectées
    }

    pannes_trouvees = []  # Liste temporaire pour stocker toutes les pannes détectées

    for panne, variables in pannes.items():
        if not all(var in data for var in variables):
            resultat["diagnostic_complet"][panne] = "variables manquantes"
            continue

        try:
            X = [[float(data[var]) for var in variables]]
            model_path = os.path.join(model_dir, f"{panne}.pkl")

            if not os.path.exists(model_path):
                resultat["diagnostic_complet"][panne] = "modèle introuvable"
                continue

            model = joblib.load(model_path)
            prediction = model.predict(X)[0]
            resultat["diagnostic_complet"][panne] = int(prediction)

            if prediction == 1:
                # Variable dominante
                importances = getattr(model, "feature_importances_", None)
                if importances is not None:
                    index_max = importances.argmax()
                    variable_dominante = variables[index_max]
                else:
                    variable_dominante = "Non disponible"

                # Score de probabilité
                if hasattr(model, "predict_proba"):
                    score = round(model.predict_proba(X)[0][1] * 100, 2)
                else:
                    score = "Non disponible"

                # Stocker les informations de la panne détectée
                panne_info = {
                    "panne": panne,
                    "variable_dominante": variable_dominante,
                    "score": score
                }
                pannes_trouvees.append(panne_info)

                log_action("prediction_detected", {
                    "panne": panne,
                    "score": score,
                    "variable_dominante": variable_dominante
                })
                # SUPPRIMER LE BREAK POUR CONTINUER À VÉRIFIER TOUTES LES PANNES

        except Exception as e:
            resultat["diagnostic_complet"][panne] = f"erreur: {str(e)}"
            log_action("error", {"panne": panne, "error": str(e)})

    # Après avoir parcouru toutes les pannes, déterminer la panne principale
    if pannes_trouvees:
        # Trier par score de confiance (le plus élevé en premier)
        pannes_trouvees.sort(key=lambda x: x["score"] if isinstance(x["score"], (int, float)) else 0, reverse=True)
        
        # Prendre la panne avec le score le plus élevé comme panne principale
        panne_principale = pannes_trouvees[0]
        resultat["panne_detectée"] = panne_principale["panne"]
        resultat["variable_dominante"] = panne_principale["variable_dominante"]
        resultat["score"] = panne_principale["score"]
        resultat["pannes_detectees"] = pannes_trouvees  # Toutes les pannes détectées

    return jsonify(resultat)

@app.route('/retrain', methods=['POST'])
def retrain():
    """Réentraîne tous les modèles avec le dataset mis à jour"""
    try:
        data = request.get_json(force=True)
        dataset_path = data.get('dataset_path', '/tmp/dataset_apprentissage.csv')
        compteur = data.get('compteur', 0)
        
        log_action("retraining_start", {"compteur": compteur, "dataset": dataset_path})
        
        # Charger le dataset
        if not os.path.exists(dataset_path):
            return jsonify({
                "success": False,
                "error": "Dataset introuvable",
                "path": dataset_path
            }), 404
        
        df = pd.read_csv(dataset_path, names=[
            'Température', 'Pression_BP', 'Pression_HP', 'Courant',
            'Tension', 'Humidité', 'Débit_air', 'Vibration',
            'Label', 'Type_Panne', 'Timestamp'
        ])
        
        resultats_entrainement = {}
        
        # Réentraîner chaque modèle
        for panne, variables in pannes.items():
            try:
                # Filtrer les données pour cette panne
                df_panne = df[df['Type_Panne'] == panne].copy()
                
                if len(df_panne) < 50:  # Minimum 50 exemples
                    resultats_entrainement[panne] = {
                        "status": "skipped",
                        "reason": f"Pas assez de données ({len(df_panne)} exemples)"
                    }
                    continue
                
                # Préparer les données
                X = df_panne[variables].values
                y = df_panne['Label'].values
                
                # Split train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Entraîner le modèle
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                
                # Évaluer
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Sauvegarder le modèle
                model_path = os.path.join(model_dir, f"{panne}.pkl")
                joblib.dump(model, model_path)
                
                resultats_entrainement[panne] = {
                    "status": "success",
                    "accuracy": round(accuracy * 100, 2),
                    "training_samples": len(X_train),
                    "test_samples": len(X_test)
                }
                
            except Exception as e:
                resultats_entrainement[panne] = {
                    "status": "error",
                    "error": str(e)
                }
        
        log_action("retraining_complete", {
            "compteur": compteur,
            "resultats": resultats_entrainement
        })
        
        return jsonify({
            "success": True,
            "compteur": compteur,
            "resultats": resultats_entrainement,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        log_action("retraining_error", {"error": str(e)})
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/train_new_fault', methods=['POST'])
def train_new_fault():
    """Entraîne un nouveau modèle pour une panne inconnue"""
    try:
        data = request.get_json(force=True)
        fault_signature = data.get('fault_signature')
        dataset_content = data.get('dataset_content')
        sample_count = data.get('sample_count', 0)
        
        if not fault_signature or not dataset_content:
            return jsonify({
                "success": False,
                "error": "Paramètres manquants"
            }), 400
        
        log_action("new_fault_training", {
            "signature": fault_signature,
            "samples": sample_count
        })
        
        # Sauvegarder le dataset
        dataset_path = os.path.join(dataset_dir, f"dataset_{fault_signature}.csv")
        with open(dataset_path, 'w') as f:
            f.write(dataset_content)
        
        # Charger et préparer les données
        df = pd.read_csv(dataset_path)
        
        # Toutes les variables pour les nouvelles pannes
        all_variables = ['Température', 'Pression_BP', 'Pression_HP', 'Courant',
                        'Tension', 'Humidité', 'Débit_air', 'Vibration']
        
        X = df[all_variables].values
        y = df['Label'].values
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Entraîner le modèle
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Évaluer
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Sauvegarder le modèle
        model_path = os.path.join(model_dir, f"{fault_signature}.pkl")
        joblib.dump(model, model_path)
        
        # Ajouter la nouvelle panne au dictionnaire
        pannes[fault_signature] = all_variables
        
        # Sauvegarder la configuration mise à jour
        config_path = os.path.join(model_dir, "pannes_config.json")
        with open(config_path, 'w') as f:
            json.dump(pannes, f, indent=2)
        
        log_action("new_fault_trained", {
            "signature": fault_signature,
            "accuracy": accuracy,
            "samples": sample_count
        })
        
        return jsonify({
            "success": True,
            "fault_signature": fault_signature,
            "accuracy": round(accuracy * 100, 2),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "model_path": model_path,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        log_action("new_fault_error", {"error": str(e)})
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/status', methods=['GET'])
def status():
    """Retourne le statut de l'agent IA"""
    models_count = len([f for f in os.listdir(model_dir) if f.endswith('.pkl')])
    
    return jsonify({
        "status": "online",
        "models_loaded": models_count,
        "known_faults": len(pannes),
        "fault_types": list(pannes.keys()),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/metrics', methods=['GET'])
def metrics():
    """Retourne les métriques de performance"""
    try:
        metrics_data = {
            "models": {},
            "total_models": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        for panne in pannes.keys():
            model_path = os.path.join(model_dir, f"{panne}.pkl")
            if os.path.exists(model_path):
                model_size = os.path.getsize(model_path)
                metrics_data["models"][panne] = {
                    "exists": True,
                    "size_kb": round(model_size / 1024, 2),
                    "last_modified": datetime.fromtimestamp(
                        os.path.getmtime(model_path)
                    ).isoformat()
                }
                metrics_data["total_models"] += 1
            else:
                metrics_data["models"][panne] = {"exists": False}
        
        return jsonify(metrics_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Charger la configuration des pannes si elle existe
    config_path = os.path.join(model_dir, "pannes_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            pannes = json.load(f)
    
    log_action("agent_start", {"pannes_count": len(pannes)})
    app.run(host='0.0.0.0', port=5000, debug=False)