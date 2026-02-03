#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transcript Utilities Module

This module provides functions for extracting transcripts from video audio:
- Extracting audio from videos
- Processing audio with Whisper model
- Aligning transcripts with video annotations
"""

import os
import tempfile
import subprocess
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Any, Optional
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_audio_from_video(
    video_path: str,
    output_path: Optional[str] = None,
    keep_audio: bool = False
) -> str:
    """
    Extract audio from a video file.
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the extracted audio (optional)
        keep_audio: Whether to keep the extracted audio file
        
    Returns:
        str: Path to the extracted audio file
    """
    # Generate output path if not provided
    if not output_path:
        video_filename = os.path.basename(video_path)
        video_name = os.path.splitext(video_filename)[0]
        
        if keep_audio:
            # Save in the same directory as the video
            audio_dir = os.path.dirname(video_path)
            output_path = os.path.join(audio_dir, f"{video_name}_audio.wav")
        else:
            # Create a temporary file
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"{video_name}_audio.wav")
    
    # Extract audio using ffmpeg
    try:
        logger.info(f"Extracting audio from {video_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Run ffmpeg command
        command = [
            "ffmpeg",
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit little-endian audio
            "-ar", "16000",  # 16kHz sample rate (optimal for Whisper)
            "-ac", "1",  # Mono
            "-y",  # Overwrite output file if it exists
            output_path
        ]
        
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Audio extracted to {output_path}")
        
        return output_path
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting audio: {str(e)}")
        return ""
    except Exception as e:
        logger.error(f"Unexpected error extracting audio: {str(e)}")
        return ""


def process_audio_with_whisper(
    audio_path: str,
    model_dict: Dict,
    chunk_size: int = 30,
    overlap: int = 5,
    language: str = "en",
    device: str = "cpu"
) -> List[Dict]:
    """
    Process audio with Whisper model to generate transcript.
    
    Args:
        audio_path: Path to the audio file
        model_dict: Dictionary containing the Whisper model and processor
        chunk_size: Size of audio chunks to process (in seconds)
        overlap: Overlap between chunks (in seconds)
        language: Language code for transcript generation
        device: Device to use for inference
        
    Returns:
        list: List of transcript segments with timestamps
    """
    if not model_dict:
        logger.error("Whisper model not provided")
        return []
    
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return []
    
    model = model_dict["model"]
    processor = model_dict["processor"]
    
    try:
        # Load audio file
        logger.info(f"Loading audio file: {audio_path}")
        
        # Use torchaudio to load the audio file
        import torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Convert to numpy array
        audio_array = waveform.squeeze().numpy()
        
        # Calculate chunk size and overlap in samples
        chunk_samples = chunk_size * sample_rate
        overlap_samples = overlap * sample_rate
        
        # Process audio in chunks
        transcript_segments = []
        
        # Calculate number of chunks
        total_samples = len(audio_array)
        num_chunks = max(1, int((total_samples - overlap_samples) / (chunk_samples - overlap_samples)))
        
        logger.info(f"Processing audio in {num_chunks} chunks")
        
        for i in tqdm(range(num_chunks), desc="Generating transcript"):
            # Calculate chunk start and end
            start_sample = int(i * (chunk_samples - overlap_samples))
            end_sample = min(start_sample + chunk_samples, total_samples)
            
            # Extract chunk
            chunk = audio_array[start_sample:end_sample]
            
            # Calculate timestamps
            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate
            
            # Process with Whisper
            inputs = processor(
                chunk,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                # Generate transcript
                result = model.generate(
                    inputs.input_features,
                    language=language,
                    task="transcribe"
                )
                
                # Decode transcript
                transcript = processor.batch_decode(
                    result,
                    skip_special_tokens=True
                )[0]
            
            # Add to segments if not empty
            if transcript.strip():
                segment = {
                    "start": start_time,
                    "end": end_time,
                    "text": transcript.strip()
                }
                transcript_segments.append(segment)
        
        logger.info(f"Generated {len(transcript_segments)} transcript segments")
        return transcript_segments
    
    except Exception as e:
        logger.error(f"Error processing audio with Whisper: {str(e)}")
        return []


def extract_transcript_from_video(
    video_path: str,
    model_dict: Dict,
    chunk_size: int = 30,
    overlap: int = 5,
    language: str = "en",
    device: str = "cpu",
    keep_audio: bool = False
) -> List[Dict]:
    """
    Extract transcript from a video file.
    
    Args:
        video_path: Path to the video file
        model_dict: Dictionary containing the Whisper model and processor
        chunk_size: Size of audio chunks to process (in seconds)
        overlap: Overlap between chunks (in seconds)
        language: Language code for transcript generation
        device: Device to use for inference
        keep_audio: Whether to keep the extracted audio file
        
    Returns:
        list: List of transcript segments with timestamps
    """
    # Extract audio from video
    audio_path = extract_audio_from_video(video_path, keep_audio=keep_audio)
    
    if not audio_path:
        logger.error("Failed to extract audio from video")
        return []
    
    # Process audio with Whisper
    transcript_segments = process_audio_with_whisper(
        audio_path=audio_path,
        model_dict=model_dict,
        chunk_size=chunk_size,
        overlap=overlap,
        language=language,
        device=device
    )
    
    # Clean up temporary audio file if not keeping it
    if not keep_audio and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
            logger.info(f"Removed temporary audio file: {audio_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary audio file: {str(e)}")
    
    return transcript_segments


def align_transcript_with_annotations(
    transcript_data: List[Dict],
    annotations: List[List],
    sampling_rate: float = 0.46
) -> List[List]:
    """
    Align transcript segments with video annotations.
    
    Args:
        transcript_data: List of transcript segments with timestamps
        annotations: List of annotations
        sampling_rate: Sampling rate of annotations in seconds
        
    Returns:
        list: Updated annotations with transcript data
    """
    if not transcript_data or not annotations:
        logger.warning("Empty transcript data or annotations")
        return annotations
    
    # Create a list to store transcript annotations
    transcript_annotations = []
    
    # For each annotation timestamp, find the corresponding transcript
    for annotation in annotations:
        timestamp = annotation[0]  # Annotation timestamp
        
        # Find transcript segments that overlap with this timestamp
        matching_segments = []
        
        for segment in transcript_data:
            start_time = segment["start"]
            end_time = segment["end"]
            
            # Check if timestamp falls within this segment
            if start_time <= timestamp <= end_time:
                matching_segments.append(segment["text"])
        
        # If no matching segments, use empty list
        if not matching_segments:
            transcript_annotations.append([""])
        else:
            transcript_annotations.append(matching_segments)
    
    # Add transcript annotations to the existing annotations
    updated_annotations = []
    
    for i, annotation in enumerate(annotations):
        # Create a new annotation with transcript data
        updated_annotation = annotation.copy()
        
        # Add transcript data
        if i < len(transcript_annotations):
            updated_annotation.append(transcript_annotations[i])
        else:
            updated_annotation.append([""])
        
        updated_annotations.append(updated_annotation)
    
    logger.info(f"Aligned {len(transcript_data)} transcript segments with {len(annotations)} annotations")
    return updated_annotations


def get_transcript_summary(transcript_data: List[Dict]) -> Dict:
    """
    Generate a summary of transcript data.
    
    Args:
        transcript_data: List of transcript segments with timestamps
        
    Returns:
        dict: Summary statistics
    """
    if not transcript_data:
        return {"error": "Empty transcript data"}
    
    # Calculate total duration
    total_duration = 0
    for segment in transcript_data:
        duration = segment["end"] - segment["start"]
        total_duration += duration
    
    # Calculate total word count
    total_words = 0
    for segment in transcript_data:
        words = segment["text"].split()
        total_words += len(words)
    
    # Calculate words per minute
    words_per_minute = (total_words / total_duration) * 60 if total_duration > 0 else 0
    
    return {
        "segment_count": len(transcript_data),
        "total_duration": total_duration,
        "total_words": total_words,
        "words_per_minute": words_per_minute,
        "language": "en"  # Assuming English as default
    }

