# 📅 2026-06-14 — Découverte des Hallucinations ASR (Exp 5)

## 🎯 Objectif
Évaluer l'impact du prétraitement sur le bruit "babble" (foule), le scénario le plus réaliste et difficile (Cocktail Party Problem).

## 🚨 Anomalie détectée
Lors de l'analyse brute de `results/babble_noise_comparison.csv`, 6 échantillons (tous à 5dB SNR) ont affiché un **WER > 1.0 (100%)**, allant jusqu'à 59.33 (soit 5933%).

## 🧠 Analyse & Décision
- **Hypothèse** : Il ne s'agit pas d'une simple erreur de reconnaissance, mais d'**hallucinations ASR**. Face à un signal acoustique ambigu (parole cible noyée dans d'autres paroles), le décodeur de Whisper tiny "invente" des phrases fluentes mais totalement déconnectées de la référence.
- **Action corrective** : Plutôt que de fausser les moyennes, nous avons développé `scripts/analyze_babble_robust.py` pour exclure ces outliers (WER ≥ 1.0) et calculer des statistiques robustes sur les 174/180 échantillons valides.

## ✅ Résultat
- Confirmation que le filtrage classique (Wiener/Spectral) aggrave catastrophiquement les performances sur le babble (+8.12% et +18.49% de dégradation à 5dB).
- Ajout de l'Insight 10 dans `docs/insights.md` sur ce phénomène critique pour le déploiement réel.