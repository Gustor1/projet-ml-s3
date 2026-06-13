# ðŸ§ª ExpÃ©rience 1 : Baseline ASR (Whisper tiny sur audio clean)

## ðŸŽ¯ Objectif
Ã‰tablir une rÃ©fÃ©rence WER sur LibriSpeech test-clean sans aucun prÃ©traitement.

## âš™ï¸ MÃ©thodologie
- **ModÃ¨le** : Whisper tiny (39M paramÃ¨tres, CPU)
- **Dataset** : 20 fichiers LibriSpeech test-clean (parole anglaise propre)
- **MÃ©trique** : Word Error Rate (WER) via jiwer
- **Mesure** : Latence d'infÃ©rence par fichier

## ðŸ“Š RÃ©sultats
| MÃ©trique | Valeur |
|----------|--------|
| WER moyen | 18.60% |
| Latence moyenne | 1894 ms/fichier |
| Fichiers traitÃ©s | 20/20 |

## ðŸ’¡ Insight principal
Whisper tiny, bien que rapide Ã  charger (~150MB), produit un WER relativement Ã©levÃ© (18.60%) mÃªme sur de l'audio propre. Ce rÃ©sultat sert de **rÃ©fÃ©rence basse** : tout preprocessing qui rÃ©duit ce WER sera considÃ©rÃ© comme bÃ©nÃ©fique.

## âš–ï¸ Trade-off identifiÃ©
- **Avantage** : ModÃ¨le trÃ¨s lÃ©ger, dÃ©ploiement facile
- **CoÃ»t** : Latence Ã©levÃ©e (1.9s) pour du "temps rÃ©el" et WER perfectible
- **HypothÃ¨se** : Un modÃ¨le plus grand (base/small) rÃ©duirait le WER mais augmenterait la latence â†’ Ã  tester

## ðŸŽ¯ Prochaine Ã©tape
Ajouter du bruit artificiel (SNR variÃ©s) puis tester l'impact de mÃ©thodes de preprocessing (Wiener, noisereduce) sur le WER.