#!/usr/bin/env python3
"""
optimization/streaming_audio.py
Chunked Audio Processing — Role 5 Deliverable (Elio)

Implements a streaming/chunked audio loader for processing long audio files
in configurable windows. This is essential for:
    - Memory-constrained edge deployments
    - Real-time streaming scenarios
    - Processing podcasts/meetings (30min+ audio)

Each chunk is processed through the ASR pipeline independently, with overlap
regions merged using a simple deduplication strategy.

Usage:
    python optimization/streaming_audio.py --input data/emotion_samples/03-01-01-01-01-01-01.wav
    python optimization/streaming_audio.py --input data/emotion_samples/03-01-01-01-01-01-01.wav --chunk-size 5 --overlap 1
"""

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional

import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Represents a chunk of audio with timing metadata."""
    data: np.ndarray
    sample_rate: int
    start_time: float  # seconds
    end_time: float    # seconds
    chunk_index: int
    is_last: bool


class StreamingAudioLoader:
    """
    Loads and yields audio in overlapping chunks for streaming processing.

    Parameters
    ----------
    chunk_duration : float
        Duration of each chunk in seconds (default: 30.0).
    overlap_duration : float
        Duration of overlap between consecutive chunks in seconds (default: 5.0).
    target_sr : int
        Target sample rate. Audio is resampled if needed (default: 16000).
    """

    def __init__(
        self,
        chunk_duration: float = 30.0,
        overlap_duration: float = 5.0,
        target_sr: int = 16000,
    ):
        if overlap_duration >= chunk_duration:
            raise ValueError("Overlap duration must be less than chunk duration.")
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.target_sr = target_sr

    def load_chunks(self, file_path: str) -> Generator[AudioChunk, None, None]:
        """
        Generator that yields AudioChunk objects from an audio file.

        Parameters
        ----------
        file_path : str
            Path to audio file (WAV, FLAC, OGG, etc.).

        Yields
        ------
        AudioChunk
            Chunk of audio with metadata.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Read entire file (for simplicity; a production system would use sf.blocks)
        audio, sr = sf.read(str(file_path), dtype="float32")

        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != self.target_sr:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
            except ImportError:
                logger.warning(f"librosa not available for resampling. Using original SR={sr}")

        total_samples = len(audio)
        total_duration = total_samples / sr
        chunk_samples = int(self.chunk_duration * sr)
        overlap_samples = int(self.overlap_duration * sr)
        step_samples = chunk_samples - overlap_samples

        logger.info(f"Audio: {total_duration:.2f}s @ {sr}Hz | "
                     f"Chunk: {self.chunk_duration}s | Overlap: {self.overlap_duration}s")

        chunk_idx = 0
        offset = 0

        while offset < total_samples:
            end = min(offset + chunk_samples, total_samples)
            chunk_data = audio[offset:end]
            is_last = (end >= total_samples)

            yield AudioChunk(
                data=chunk_data,
                sample_rate=sr,
                start_time=round(offset / sr, 3),
                end_time=round(end / sr, 3),
                chunk_index=chunk_idx,
                is_last=is_last,
            )

            if is_last:
                break

            offset += step_samples
            chunk_idx += 1

    def get_total_chunks(self, file_path: str) -> int:
        """Estimate the total number of chunks for a given file."""
        info = sf.info(str(file_path))
        total_duration = info.duration
        step = self.chunk_duration - self.overlap_duration
        return max(1, int(np.ceil((total_duration - self.overlap_duration) / step)))


def merge_transcriptions(chunks_text: List[str], overlap_words: int = 3) -> str:
    """
    Merge overlapping chunk transcriptions by removing duplicated words
    at chunk boundaries.

    Parameters
    ----------
    chunks_text : list of str
        Ordered list of transcriptions per chunk.
    overlap_words : int
        Number of words at boundaries to check for deduplication.

    Returns
    -------
    str
        Merged transcription.
    """
    if not chunks_text:
        return ""
    if len(chunks_text) == 1:
        return chunks_text[0]

    merged = chunks_text[0]

    for i in range(1, len(chunks_text)):
        prev_words = merged.split()
        curr_words = chunks_text[i].split()

        if not prev_words or not curr_words:
            merged += " " + chunks_text[i]
            continue

        # Find best overlap match
        best_overlap = 0
        tail = prev_words[-overlap_words:] if len(prev_words) >= overlap_words else prev_words

        for k in range(min(overlap_words, len(curr_words)), 0, -1):
            if tail[-k:] == curr_words[:k]:
                best_overlap = k
                break

        if best_overlap > 0:
            merged += " " + " ".join(curr_words[best_overlap:])
        else:
            merged += " " + chunks_text[i]

    return merged.strip()


# -------------------------------------------------------------------------
# CLI demo
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Chunked audio processing demo"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to an audio file"
    )
    parser.add_argument(
        "--chunk-size", type=float, default=30.0,
        help="Chunk duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--overlap", type=float, default=5.0,
        help="Overlap duration in seconds (default: 5)"
    )
    parser.add_argument(
        "--transcribe", action="store_true",
        help="Run Whisper ASR on each chunk (requires transformers)"
    )
    args = parser.parse_args()

    loader = StreamingAudioLoader(
        chunk_duration=args.chunk_size,
        overlap_duration=args.overlap,
    )

    print(f"\n{'='*60}")
    print(f"  STREAMING AUDIO PROCESSOR")
    print(f"{'='*60}")
    print(f"  Input:      {args.input}")
    print(f"  Chunk size: {args.chunk_size}s")
    print(f"  Overlap:    {args.overlap}s")
    print(f"{'-'*60}")

    asr_pipe = None
    if args.transcribe:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading Whisper-tiny for chunked transcription...")
        asr_pipe = hf_pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            device="cpu",
        )

    transcriptions = []

    for chunk in loader.load_chunks(args.input):
        duration = chunk.end_time - chunk.start_time
        print(f"  Chunk {chunk.chunk_index}: [{chunk.start_time:.1f}s - {chunk.end_time:.1f}s] "
              f"({duration:.1f}s, {len(chunk.data)} samples)", end="")

        if asr_pipe is not None:
            t0 = time.perf_counter()
            result = asr_pipe(chunk.data, generate_kwargs={"language": "en"})
            elapsed = time.perf_counter() - t0
            text = result["text"].strip()
            transcriptions.append(text)
            print(f" -> \"{text}\" ({elapsed:.2f}s)")
        else:
            print()

    if transcriptions:
        merged = merge_transcriptions(transcriptions)
        print(f"\n  Merged transcription:")
        print(f"  \"{merged}\"")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
