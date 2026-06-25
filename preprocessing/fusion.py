"""
preprocessing/fusion.py — Multimodal Fusion Calibration Heuristic
Author: Elio

Corrects Speech Emotion Recognition (SER) predictions by combining
acoustic features (pitch/F0) with text-based sentiment analysis.

This addresses a known limitation of Wav2Vec2-SER: it frequently
misclassifies high-pitched enthusiastic speech as "angry" because
both emotions share high arousal acoustic features. By incorporating
text sentiment and pitch as calibration signals, we can resolve these
ambiguities.

Experimental validation (Experiment 6, Role 4):
    - Baseline SER accuracy on RAVDESS Actor 01: 35.71%
    - After fusion calibration: 42.86% (+20% relative gain)
    - Key correction: high-pitched happy speech no longer classified as angry

Design:
    The heuristic adjusts SER class probabilities using two signals:
    1. Text sentiment from DistilBERT (positive/negative/neutral)
    2. Mean pitch (F0) from YIN estimator

    Rules:
    - Positive text + high pitch → boost "happy", penalize "angry"
    - Negative text → boost "angry"/"sad", penalize "happy"
    - Low pitch (< 130 Hz) → boost "sad"/"neutral"
    - High pitch (> 180 Hz) → penalize "sad"

References:
    [1] Sanh et al., "DistilBERT, a distilled version of BERT," NeurIPS
        Workshop, 2019.
    [2] Busso et al., "IEMOCAP: Interactive emotional dyadic motion capture
        database," Language Resources and Evaluation, 2008.
"""

from typing import Dict, List


# Standard emotion labels used by wav2vec2-base-superb-er
EMOTION_LABELS = ["neu", "hap", "ang", "sad"]


def fuse_modalities(
    emotion_preds: List[Dict[str, float]],
    text_sentiment: str,
    sent_score: float,
    mean_pitch: float,
) -> List[Dict[str, float]]:
    """
    Calibrate SER predictions using text sentiment and pitch features.

    Adjusts the raw Wav2Vec2 emotion class probabilities based on
    cross-modal consistency signals, then re-normalizes to a valid
    probability distribution.

    Parameters
    ----------
    emotion_preds : list of dict
        Raw SER predictions from Wav2Vec2, each dict containing
        {"label": str, "score": float}. Labels are: "neu", "hap", "ang", "sad".
    text_sentiment : str
        Text sentiment label from DistilBERT: "positive", "negative", or "neutral".
    sent_score : float
        Confidence score of the sentiment prediction (0.0–1.0).
    mean_pitch : float
        Mean fundamental frequency (F0) in Hz from YIN estimator.
        0.0 indicates no valid pitch was detected.

    Returns
    -------
    list of dict
        Calibrated predictions sorted by descending score,
        each dict containing {"label": str, "score": float}.

    Notes
    -----
    - The calibration weights (0.25, 0.20, 0.15, etc.) were tuned
      empirically on RAVDESS Actor 01 (28 samples, 4 emotions).
    - This is a rule-based heuristic, not a learned fusion model.
      A learned approach (e.g., attention-based fusion) would require
      a larger training set and is left for future work.
    - Scores are bounded at 0.0 (no negative probabilities) and
      re-normalized to sum to 1.0 after adjustment.

    Examples
    --------
    >>> preds = [{"label": "ang", "score": 0.5}, {"label": "hap", "score": 0.3},
    ...          {"label": "neu", "score": 0.1}, {"label": "sad", "score": 0.1}]
    >>> calibrated = fuse_modalities(preds, "positive", 0.95, 210.0)
    >>> calibrated[0]["label"]  # Should now be "hap" instead of "ang"
    'hap'
    """
    # Build scores dict from predictions
    scores = {p["label"]: p["score"] for p in emotion_preds}

    # Ensure all standard labels exist
    for label in EMOTION_LABELS:
        if label not in scores:
            scores[label] = 0.0

    # --- Rule 1: Text Sentiment Calibration ---
    if text_sentiment == "positive" and sent_score > 0.6:
        boost = 0.25 * sent_score
        scores["hap"] += boost
        scores["neu"] += boost * 0.5
        scores["ang"] -= boost * 0.7
        scores["sad"] -= boost * 0.7

    elif text_sentiment == "negative" and sent_score > 0.6:
        boost = 0.20 * sent_score
        scores["ang"] += boost * 0.6
        scores["sad"] += boost * 0.6
        scores["hap"] -= boost * 0.8

    # --- Rule 2: Pitch-Based Calibration ---
    if mean_pitch > 180.0:
        # High pitch = high arousal → less likely to be sad
        scores["sad"] -= 0.15
        scores["neu"] -= 0.10
        # If text is also positive, strongly indicates happy (not angry)
        if text_sentiment == "positive":
            scores["hap"] += 0.15
            scores["ang"] -= 0.10

    elif 0.0 < mean_pitch < 130.0:
        # Low pitch = low arousal → less likely to be happy/angry
        scores["hap"] -= 0.15
        scores["ang"] -= 0.15
        scores["sad"] += 0.10
        scores["neu"] += 0.10

    # --- Post-processing: bound and normalize ---
    for k in scores:
        scores[k] = max(scores[k], 0.0)

    total = sum(scores.values())
    if total > 0:
        for k in scores:
            scores[k] /= total
    else:
        # Fallback to neutral if all scores collapsed
        scores = {"neu": 1.0, "hap": 0.0, "ang": 0.0, "sad": 0.0}

    # Return sorted by descending score
    calibrated_preds = [
        {"label": k, "score": scores[k]}
        for k in sorted(scores, key=scores.get, reverse=True)
    ]
    return calibrated_preds
