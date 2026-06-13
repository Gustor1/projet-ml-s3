# ️ Engineering Trade-offs

> *Ce document résume les compromis d'ingénierie identifiés lors du projet. Pour l'analyse détaillée, voir `docs/insights.md`.*

## 1. Accuracy vs. Noise Level
- **Observation** : Le prétraitement (Wiener) n'améliore le WER qu'en dessous de 10dB SNR.
- **Trade-off** : Activer le filtre "toujours" dégrade la performance sur l'audio propre.
- **Décision** : Implémenter un seuil d'activation (SNR < 12dB).

## 2. Latency vs. Complexity
- **Observation** : Le filtre Wiener ajoute ~100-200ms de latence.
- **Trade-off** : Complexité algorithmique vs temps réel.
- **Décision** : Le coût est acceptable (< 10% du temps total d'inférence) car il permet de récupérer ~5% de WER en environnement bruyant.

## 3. Model Size vs. Performance
- **Observation** : Whisper tiny (39M params) a un WER de base de ~19% sur audio propre.
- **Trade-off** : Modèle léger (déploiement facile) vs Modèle lourd (meilleure précision).
- **Décision** : Utiliser un modèle tiny pour prouver que le preprocessing est nécessaire même avec un modèle faible. Un modèle "base" ou "small" réduirait le WER mais augmenterait la latence de façon prohibitive pour du mobile.