# 📅 2026-06-20 — Rôle 3 : Intégration ASR & Analyse Cross-Modale du Sarcasme

## 🎯 Objectif
Finaliser l'intégration des modèles ASR (Whisper, Wav2Vec2), corriger les défauts du wrapper existant, et implémenter l'analyse d'ablation cross-modale pour mesurer l'impact des erreurs de transcription ASR sur la détection de sarcasme en aval.

---

## 🚨 Problèmes identifiés & Limites (Ce qui ne marchait pas)

Lors de l'audit du code initial dans le dossier `asr/`, plusieurs limitations et erreurs de conception ont été détectées :

1. **Incohérence d'héritage dans `whisper_wrapper.py`** :
   - Le wrapper n'appelait pas `super().__init__()` de la classe de base `BaseASR`.
   - Il chargeait systématiquement le modèle Whisper en CPU, ignorant l'accélération matérielle CUDA (contrairement à `wav2vec_wrapper.py`), rendant les tests extrêmement lents.
   - Aucune validation sur les langues spécifiées n'était effectuée (le projet doit supporter explicitement EN, FR et ZH).
2. **Manque de l'analyse Cross-Modale** :
   - Le script `ablation_study.py` se contentait de mesurer le WER/CER traditionnel.
   - Il n'y avait aucun pont mesurant comment une erreur acoustique ou typographique générée par Whisper se cascade et induit en erreur l'analyse de sentiment de DistilBERT, faussant ainsi les alertes de sarcasme (mismatch entre sentiment textuel et émotion vocale).

---

## 🛠️ Solutions appliquées & Choix de conception

### 1. Refactoring et Robustesse du Wrapper Whisper
- **Validation stricte** : Intégration de la constante `SUPPORTED_LANGUAGES` pour interdire toute langue non supportée (EN/FR/ZH).
- **Accélération matérielle** : Détection automatique de CUDA pour basculer de manière transparente sur GPU si disponible, réduisant drastiquement le temps d'inférence en batch.
- **Héritage propre** : Correction de l'appel au constructeur de `BaseASR` pour maintenir l'uniformité des API de notre framework d'évaluation.

### 2. Création du pipeline d'étude d'ablation cross-modale (`cross_modal_ablation.py`)
Nous avons conçu un script d'évaluation de bout en bout qui simule la chaîne multimodale complète :
1. Charger les fichiers audio et pré-calculer les vecteurs d'émotions vocales via `Wav2Vec2-SER` ainsi que le sentiment de référence du texte "ground truth" (via DistilBERT).
2. Pour chaque taille de modèle Whisper (tiny, base, small) :
   - Générer la transcription.
   - Mesurer le **WER / CER** traditionnel.
   - Soumettre cette transcription ASR à DistilBERT pour obtenir le sentiment prédit.
   - Détecter le sarcasme via la logique de mismatch (ex: mots positifs prononcés sur un ton colérique/triste).
   - Calculer les métriques clés de cascade d'erreurs :
     - **Sentiment Flip Rate** : % de fois où l'erreur de transcription fait changer la classe de sentiment (ex: "I'm fine" compris comme "I fail" changeant le sentiment de positif à négatif).
     - **Sarcasm False Positive Rate** : % de faux sarcasmes détectés suite à une mauvaise transcription.
     - **Sarcasm False Negative Rate** : % de vrais sarcasmes manqués suite à une erreur ASR.
     - **Agreement Rate** : % de concordance globale des prédictions de sarcasme entre la transcription ASR et le texte parfait de référence.

---

## 📈 Résultats & Observations (Insights Humains)

- **Propagation des erreurs** : Les modèles Whisper plus petits (comme `tiny`) génèrent plus de fautes de frappe ou omettent des négations (ex: "cannot" devenant "can not" ou "can"). Cela provoque un **Sentiment Flip Rate** élevé, qui fausse directement le verdict de sarcasme.
- **Robustesse du sarcasme** : L'utilisation de modèles plus grands (`base` ou `small`) diminue significativement le taux de faux positifs pour le sarcasme, confirmant l'hypothèse de départ : **la qualité de l'ASR est un prérequis indispensable pour l'analyse de sentiments multimodale fiable**.
- Les résultats détaillés et les résumés sont exportés automatiquement sous format CSV (`results/cross_modal_ablation.csv` et `results/cross_modal_ablation_summary.csv`) pour être exploitables par les autres membres du groupe pour le rapport scientifique final.
