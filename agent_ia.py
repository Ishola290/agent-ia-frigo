from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pickle
import os
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

app = Flask(__name__)
CORS(app)
# Fonction de diagnostic pour Render.com
def diagnose_filesystem():
    """Diagnostique les problèmes de filesystem"""
    print("=" * 50)
    print("🔍 DIAGNOSTIC FILESYSTEM")
    print("=" * 50)
    
    app_dir = os.path.dirname(__file__)
    print(f"📁 Répertoire de l'application: {app_dir}")
    print(f"📁 Répertoire de travail: {os.getcwd()}")
    print(f"🔐 UID/GID: {os.getuid()}/{os.getgid()}" if hasattr(os, 'getuid') else "🔐 Windows")
    
    # Vérifier /app
    if os.path.exists('/app'):
        print(f"📦 /app existe")
        try:
            items = os.listdir('/app')
            print(f"   Contenu: {items[:10]}")  # Premiers 10 items
            for item in ['models', 'datasets', 'logs']:
                path = f'/app/{item}'
                if os.path.exists(path):
                    is_dir = os.path.isdir(path)
                    print(f"   ⚠️  {item}: {'DOSSIER' if is_dir else 'FICHIER (PROBLÈME!)'}")
        except Exception as e:
            print(f"   ❌ Erreur lecture: {e}")
    
    print("=" * 50)

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

# Créer les dossiers s'ils n'existent pas - avec gestion d'erreur robuste
for directory in [model_dir, dataset_dir, log_dir]:
    try:
        if not os.path.exists(directory):
            os.makedirs(directory, mode=0o755)
        elif not os.path.isdir(directory):
            # Si c'est un fichier au lieu d'un dossier, le supprimer
            os.remove(directory)
            os.makedirs(directory, mode=0o755)
    except (FileExistsError, PermissionError, OSError) as e:
        # Si le dossier existe déjà ou erreur de permission, continuer
        print(f"Note: Dossier {directory} - {str(e)}")
        pass

def load_model(model_path):
    """Charge un modèle quel que soit son format (.pkl, .joblib, etc.)"""
    if not os.path.exists(model_path):
        return None
    
    try:
        # Essayer joblib en premier (supporte plus de formats)
        model = joblib.load(model_path)
        return model
    except Exception as e1:
        try:
            # Essayer pickle en fallback
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e2:
            raise Exception(f"Impossible de charger le modèle {model_path}. Joblib: {str(e1)}, Pickle: {str(e2)}")

def log_action(action, details):
    """Enregistre les actions dans un fichier de log"""
    try:
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "action": action,
            "details": details
        }
        
        log_file = os.path.join(log_dir, f"agent_ia_{datetime.now().strftime('%Y%m%d')}.log")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Erreur lors du logging: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction standard - Détection multi-pannes améliorée"""
    try:
        data = request.get_json(force=True)
        
        resultat = {
            "pannes_detectees": [],  # Liste de toutes les pannes détectées (prédiction = 1)
            "panne_detectee": None,  # Panne principale (rétrocompatibilité)
            "variable_dominante": None,
            "score": None,
            "diagnostic_complet": {},  # Résultat brut de chaque modèle (0 ou 1)
            "modeles_charges": {},  # Info sur les modèles utilisés
            "avertissement": None,
            "timestamp": datetime.now().isoformat()
        }

        pannes_detectees_temp = []

        for panne, variables in pannes.items():
            # Vérifier que toutes les variables requises sont présentes
            variables_manquantes = [var for var in variables if var not in data]
            
            if variables_manquantes:
                resultat["diagnostic_complet"][panne] = 0
                resultat["modeles_charges"][panne] = {
                    "statut": "variables_manquantes",
                    "variables_manquantes": variables_manquantes
                }
                continue

            try:
                # Préparer les données d'entrée
                X = [[float(data[var]) for var in variables]]
                
                # Chercher le modèle dans différents formats et emplacements
                model_paths = [
                    os.path.join(model_dir, f"{panne}.pkl"),
                    os.path.join(model_dir, f"{panne}.joblib"),
                    os.path.join(model_dir, f"model_{panne}.pkl"),
                    os.path.join(model_dir, f"model_{panne}.joblib")
                ]
                
                model = None
                model_path_used = None
                model_format = None
                
                for model_path in model_paths:
                    if os.path.exists(model_path):
                        try:
                            model = load_model(model_path)
                            model_path_used = model_path
                            model_format = "joblib" if model_path.endswith('.joblib') else "pkl"
                            break
                        except Exception as e:
                            print(f"Erreur chargement {model_path}: {str(e)}")
                            continue
                
                if model is None:
                    resultat["diagnostic_complet"][panne] = 0
                    resultat["modeles_charges"][panne] = {
                        "statut": "modele_introuvable",
                        "chemins_testes": model_paths
                    }
                    continue

                # Faire la prédiction
                prediction = int(model.predict(X)[0])
                resultat["diagnostic_complet"][panne] = prediction
                
                resultat["modeles_charges"][panne] = {
                    "chemin": os.path.basename(model_path_used),
                    "format": model_format
                }

                # Si panne détectée (prédiction = 1)
                if prediction == 1:
                    # Extraire la variable dominante
                    importances = getattr(model, "feature_importances_", None)
                    if importances is not None and len(importances) > 0:
                        index_max = importances.argmax()
                        variable_dominante = variables[index_max]
                    else:
                        variable_dominante = "Non disponible"

                    # Calculer le score de confiance
                    if hasattr(model, "predict_proba"):
                        try:
                            proba = model.predict_proba(X)[0]
                            # Vérifier que proba a au moins 2 éléments (classe 0 et classe 1)
                            if len(proba) > 1:
                                score = round(proba[1] * 100, 2)
                            else:
                                score = 100.0
                        except Exception as e:
                            print(f"Erreur predict_proba pour {panne}: {str(e)}")
                            score = 100.0
                    else:
                        score = 100.0  # Valeur par défaut si predict_proba non disponible

                    panne_info = {
                        "panne": panne,
                        "variable_dominante": variable_dominante,
                        "score": score,
                        "modele_utilise": os.path.basename(model_path_used),
                        "variables_analysees": variables
                    }
                    
                    pannes_detectees_temp.append(panne_info)
                    
                    log_action("panne_detectee", {
                        "panne": panne,
                        "score": score,
                        "variable_dominante": variable_dominante
                    })

            except Exception as e:
                resultat["diagnostic_complet"][panne] = 0
                resultat["modeles_charges"][panne] = {
                    "statut": "erreur",
                    "erreur": str(e)
                }
                log_action("erreur_prediction", {"panne": panne, "error": str(e)})

        # Trier les pannes détectées par score (du plus élevé au plus bas)
        pannes_detectees_temp.sort(key=lambda x: x["score"], reverse=True)
        
        # Mettre à jour le résultat final
        resultat["pannes_detectees"] = pannes_detectees_temp
        
        # Rétrocompatibilité : panne_detectee et autres champs
        if pannes_detectees_temp:
            resultat["panne_detectee"] = pannes_detectees_temp[0]["panne"]
            resultat["variable_dominante"] = pannes_detectees_temp[0]["variable_dominante"]
            resultat["score"] = pannes_detectees_temp[0]["score"]
            
            # Avertissement si pannes multiples
            if len(pannes_detectees_temp) > 1:
                resultat["avertissement"] = f"{len(pannes_detectees_temp)} pannes détectées simultanément. Analyse de corrélation recommandée."
        else:
            resultat["panne_detectee"] = None
            resultat["variable_dominante"] = "Non disponible"
            resultat["score"] = 0

        log_action("prediction_complete", {
            "nombre_pannes": len(pannes_detectees_temp),
            "pannes": [p["panne"] for p in pannes_detectees_temp]
        })

        return jsonify(resultat)
    
    except Exception as e:
        log_action("erreur_endpoint_predict", {"error": str(e)})
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/predict_advanced', methods=['POST'])
def predict_advanced():
    """
    Pipeline complet avec :
    1. Isolation Forest → Détection d'anomalies (nouvelles pannes)
    2. 12 modèles individuels → Prédiction des pannes connues
    3. Multi-label Random Forest → Validation des prédictions (informatif)
    """
    try:
        data = request.get_json(force=True)
        
        resultat = {
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "pannes_detectees": [],
            "panne_detectee": None,
            "variable_dominante": None,
            "score": None,
            "diagnostic_complet": {},
            "multi_label_validation": {},
            "avertissement": None,
            "timestamp": datetime.now().isoformat()
        }

        # ÉTAPE 1 : Isolation Forest - Détection d'anomalies
        try:
            isolation_model_path = os.path.join(model_dir, "isolation_forest_model.joblib")
            if os.path.exists(isolation_model_path):
                isolation_model = load_model(isolation_model_path)
                
                # Toutes les variables pour l'isolation forest
                all_variables = ['Température', 'Pression_BP', 'Pression_HP', 'Courant',
                                'Tension', 'Humidité', 'Débit_air', 'Vibration']
                X = [[float(data.get(var, 0)) for var in all_variables]]
                
                # Prédiction : -1 = anomalie, 1 = normal
                anomaly_pred = int(isolation_model.predict(X)[0])
                anomaly_score = float(isolation_model.score_samples(X)[0])
                
                resultat["is_anomaly"] = (anomaly_pred == -1)
                resultat["anomaly_score"] = round(anomaly_score, 4)
                
                log_action("isolation_forest_check", {
                    "is_anomaly": resultat["is_anomaly"],
                    "anomaly_score": resultat["anomaly_score"]
                })
        except Exception as e:
            print(f"Erreur Isolation Forest: {str(e)}")
            resultat["is_anomaly"] = False

        # ÉTAPE 2 : 12 modèles individuels (Source de vérité principale)
        pannes_detectees_temp = []
        for panne, variables in pannes.items():
            variables_manquantes = [var for var in variables if var not in data]
            if variables_manquantes:
                resultat["diagnostic_complet"][panne] = 0
                continue

            try:
                X = [[float(data[var]) for var in variables]]
                
                model_paths = [
                    os.path.join(model_dir, f"model_{panne}.joblib"),
                    os.path.join(model_dir, f"{panne}.joblib"),
                ]
                
                model = None
                for model_path in model_paths:
                    if os.path.exists(model_path):
                        model = load_model(model_path)
                        break
                
                if model is None:
                    resultat["diagnostic_complet"][panne] = 0
                    continue

                prediction = int(model.predict(X)[0])
                resultat["diagnostic_complet"][panne] = prediction

                if prediction == 1:
                    importances = getattr(model, "feature_importances_", None)
                    if importances is not None and len(importances) > 0:
                        index_max = importances.argmax()
                        variable_dominante = variables[index_max]
                    else:
                        variable_dominante = "Non disponible"

                    if hasattr(model, "predict_proba"):
                        try:
                            proba = model.predict_proba(X)[0]
                            if len(proba) > 1:
                                score = round(proba[1] * 100, 2)
                            else:
                                score = 100.0
                        except:
                            score = 100.0
                    else:
                        score = 100.0

                    panne_info = {
                        "panne": panne,
                        "variable_dominante": variable_dominante,
                        "score": score,
                        "variables_analysees": variables
                    }
                    pannes_detectees_temp.append(panne_info)

            except Exception as e:
                resultat["diagnostic_complet"][panne] = 0

        # ÉTAPE 3 : Multi-label Random Forest - Validation (Informatif uniquement)
        try:
            multi_label_path = os.path.join(model_dir, "multi_label_random_forest_model.joblib")
            if os.path.exists(multi_label_path):
                multi_label_model = load_model(multi_label_path)
                
                all_variables = ['Température', 'Pression_BP', 'Pression_HP', 'Courant',
                                'Tension', 'Humidité', 'Débit_air', 'Vibration']
                X = [[float(data.get(var, 0)) for var in all_variables]]
                
                # Prédiction multi-label (retourne un tableau de 0/1 pour chaque panne)
                multi_pred = multi_label_model.predict(X)[0]
                
                # Mapping des pannes aux indices (ordre alphabétique)
                pannes_list = sorted(pannes.keys())
                for i, panne in enumerate(pannes_list):
                    if i < len(multi_pred):
                        resultat["multi_label_validation"][panne] = int(multi_pred[i])
                
                # CORRECTION : On ne filtre PLUS la liste pannes_detectees_temp.
                # Le modèle multi-label peut avoir un ordre de classes différent lors de l'entraînement,
                # ce qui effaçait toutes les détections. On garde les 12 modèles spécialisés.
                
                log_action("multi_label_validation", {
                    "pannes_detectees": len(pannes_detectees_temp),
                    "validation": resultat["multi_label_validation"]
                })
        except Exception as e:
            print(f"Erreur Multi-label: {str(e)}")

        # Résultat final
        pannes_detectees_temp.sort(key=lambda x: x["score"], reverse=True)
        resultat["pannes_detectees"] = pannes_detectees_temp
        
        if pannes_detectees_temp:
            resultat["panne_detectee"] = pannes_detectees_temp[0]["panne"]
            resultat["variable_dominante"] = pannes_detectees_temp[0]["variable_dominante"]
            resultat["score"] = pannes_detectees_temp[0]["score"]
            
            if len(pannes_detectees_temp) > 1:
                resultat["avertissement"] = f"{len(pannes_detectees_temp)} pannes détectées simultanément."
        else:
            resultat["panne_detectee"] = None
            resultat["variable_dominante"] = "Non disponible"
            resultat["score"] = 0

        # Si anomalie détectée mais aucune panne connue → nouvelle panne
        if resultat["is_anomaly"] and not pannes_detectees_temp:
            resultat["avertissement"] = "ANOMALIE DÉTECTÉE : Nouvelle panne potentielle non répertoriée !"

        log_action("prediction_advanced_complete", {
            "is_anomaly": resultat["is_anomaly"],
            "nombre_pannes": len(pannes_detectees_temp),
            "pannes": [p["panne"] for p in pannes_detectees_temp]
        })

        return jsonify(resultat)
    
    except Exception as e:
        log_action("erreur_endpoint_predict_advanced", {"error": str(e)})
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Réentraîne tous les modèles avec le dataset mis à jour"""
    try:
        data = request.get_json(force=True)
        dataset_path = data.get('dataset_path', './dataset_apprentissage.csv')
        compteur = data.get('compteur', 0)
        
        log_action("retraining_start", {"compteur": compteur, "dataset": dataset_path})
        
        # Chercher le dataset dans plusieurs emplacements
        possible_paths = [
            dataset_path,
            os.path.join(dataset_dir, 'dataset_apprentissage.csv'),
            './dataset_apprentissage.csv',
            '/tmp/dataset_apprentissage.csv'
        ]
        
        dataset_found = None
        for path in possible_paths:
            if os.path.exists(path):
                dataset_found = path
                break
        
        if dataset_found is None:
            return jsonify({
                "success": False,
                "error": "Dataset introuvable",
                "chemins_testes": possible_paths
            }), 404
        
        # Charger le dataset
        df = pd.read_csv(dataset_found)
        
        # Si le CSV n'a pas de header, définir les colonnes
        if df.shape[1] == 11 and df.columns[0] != 'Température':
            df.columns = [
                'Température', 'Pression_BP', 'Pression_HP', 'Courant',
                'Tension', 'Humidité', 'Débit_air', 'Vibration',
                'Label', 'Type_Panne', 'Timestamp'
            ]
        
        resultats_entrainement = {}
        
        # Réentraîner chaque modèle
        for panne, variables in pannes.items():
            try:
                # Filtrer les données pour cette panne
                df_panne = df[df['Type_Panne'] == panne].copy()
                
                if len(df_panne) < 50:  # Minimum 50 exemples
                    resultats_entrainement[panne] = {
                        "status": "skipped",
                        "reason": f"Pas assez de données ({len(df_panne)} exemples, minimum 50 requis)"
                    }
                    continue
                
                # Préparer les données
                X = df_panne[variables].values
                y = df_panne['Label'].values
                
                # Vérifier qu'il y a au moins 2 classes
                if len(set(y)) < 2:
                    resultats_entrainement[panne] = {
                        "status": "skipped",
                        "reason": "Dataset ne contient qu'une seule classe"
                    }
                    continue
                
                # Split train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
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
                
                # Sauvegarder le modèle en .joblib (format recommandé)
                model_path = os.path.join(model_dir, f"model_{panne}.joblib")
                joblib.dump(model, model_path)
                
                resultats_entrainement[panne] = {
                    "status": "success",
                    "accuracy": round(accuracy * 100, 2),
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "model_format": "joblib",
                    "model_path": model_path
                }
                
                log_action("model_retrained", {
                    "panne": panne,
                    "accuracy": accuracy,
                    "samples": len(X_train)
                })
                
            except Exception as e:
                resultats_entrainement[panne] = {
                    "status": "error",
                    "error": str(e)
                }
                log_action("retraining_error", {"panne": panne, "error": str(e)})
        
        log_action("retraining_complete", {
            "compteur": compteur,
            "resultats": resultats_entrainement
        })
        
        return jsonify({
            "success": True,
            "compteur": compteur,
            "resultats": resultats_entrainement,
            "dataset_utilise": dataset_found,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        log_action("retraining_error_global", {"error": str(e)})
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
                "error": "Paramètres manquants (fault_signature et dataset_content requis)"
            }), 400
        
        log_action("new_fault_training_start", {
            "signature": fault_signature,
            "samples": sample_count
        })
        
        # Sauvegarder le dataset
        dataset_path = os.path.join(dataset_dir, f"dataset_{fault_signature}.csv")
        with open(dataset_path, 'w', encoding='utf-8') as f:
            f.write(dataset_content)
        
        # Charger et préparer les données
        df = pd.read_csv(dataset_path)
        
        # Toutes les variables pour les nouvelles pannes
        all_variables = ['Température', 'Pression_BP', 'Pression_HP', 'Courant',
                        'Tension', 'Humidité', 'Débit_air', 'Vibration']
        
        # Vérifier que toutes les colonnes existent
        missing_cols = [col for col in all_variables if col not in df.columns]
        if missing_cols:
            return jsonify({
                "success": False,
                "error": f"Colonnes manquantes dans le dataset: {missing_cols}"
            }), 400
        
        X = df[all_variables].values
        y = df['Label'].values
        
        # Vérifier qu'il y a assez de données
        if len(X) < 50:
            return jsonify({
                "success": False,
                "error": f"Pas assez de données: {len(X)} exemples (minimum 50 requis)"
            }), 400
        
        # Vérifier qu'il y a au moins 2 classes
        if len(set(y)) < 2:
            return jsonify({
                "success": False,
                "error": "Le dataset ne contient qu'une seule classe"
            }), 400
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
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
        model_path = os.path.join(model_dir, f"{fault_signature}.joblib")
        joblib.dump(model, model_path)
        
        # Ajouter la nouvelle panne au dictionnaire
        pannes[fault_signature] = all_variables
        
        # Sauvegarder la configuration mise à jour
        config_path = os.path.join(model_dir, "pannes_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(pannes, f, indent=2, ensure_ascii=False)
        
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
    try:
        model_files = []
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith(('.pkl', '.joblib'))]
        
        models_count = len(model_files)
        
        return jsonify({
            "status": "online",
            "models_loaded": models_count,
            "known_faults": len(pannes),
            "fault_types": list(pannes.keys()),
            "model_files": model_files,
            "model_directory": model_dir,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

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
            # Chercher le modèle dans différents formats
            model_found = False
            for prefix in ['', 'model_']:
                for ext in ['.pkl', '.joblib']:
                    model_path = os.path.join(model_dir, f"{prefix}{panne}{ext}")
                    if os.path.exists(model_path):
                        model_size = os.path.getsize(model_path)
                        metrics_data["models"][panne] = {
                            "exists": True,
                            "format": ext.replace('.', ''),
                            "size_kb": round(model_size / 1024, 2),
                            "path": os.path.basename(model_path),
                            "last_modified": datetime.fromtimestamp(
                                os.path.getmtime(model_path)
                            ).isoformat()
                        }
                        metrics_data["total_models"] += 1
                        model_found = True
                        break
                if model_found:
                    break
            
            if not model_found:
                metrics_data["models"][panne] = {"exists": False}
        
        return jsonify(metrics_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de health check"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Diagnostic au démarrage
    diagnose_filesystem()
    
    # Charger la configuration des pannes si elle existe
    config_path = os.path.join(model_dir, "pannes_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_pannes = json.load(f)
                pannes.update(loaded_pannes)
                print(f"✅ Configuration chargée: {len(pannes)} types de pannes")
        except Exception as e:
            print(f"⚠️  Erreur lors du chargement de la configuration: {str(e)}")
    
    log_action("agent_start", {
        "pannes_count": len(pannes),
        "model_dir": model_dir
    })
    
    print(f"🚀 Agent IA démarré")
    print(f"📁 Répertoire modèles: {model_dir}")
    print(f"🔍 Types de pannes connus: {len(pannes)}")
    
    app.run(host='0.0.0.0', port=10000, debug=False)
