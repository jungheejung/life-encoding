#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Utilities Module

This module provides functions for video processing, including:
- Video validation and metadata extraction
- Frame extraction at precise intervals
- Video preprocessing for model input
"""

import os
import cv2
import numpy as np
import av  # Use PyAV instead of decord
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_input(video_path: str) -> bool:
    """
    Validate that the input video file exists and can be opened.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        bool: True if the video is valid, False otherwise
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False
    
    try:
        # Try opening with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video with OpenCV: {video_path}")
            cap.release()
            
            # Try with PyAV as a fallback
            try:
                container = av.open(video_path)
                logger.info(f"Successfully opened video with PyAV: {video_path}")
                container.close()
                return True
            except Exception as e:
                logger.error(f"Could not open video with PyAV: {str(e)}")
                return False
        cap.release()
        return True
        
    except Exception as e:
        logger.error(f"Error validating video: {str(e)}")
        return False


def get_video_metadata(video_path: str) -> Dict:
    """
    Extract metadata from a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        dict: Dictionary containing video metadata
    """
    metadata = {}
    
    try:
        # Try with OpenCV first
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            # Get basic metadata
            metadata['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            metadata['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            metadata['fps'] = cap.get(cv2.CAP_PROP_FPS)
            metadata['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            metadata['duration'] = metadata['frame_count'] / metadata['fps'] if metadata['fps'] > 0 else 0
            cap.release()
        else:
            # Fallback to PyAV
            container = av.open(video_path)
            video_stream = next(s for s in container.streams if s.type == 'video')
            metadata['width'] = video_stream.width
            metadata['height'] = video_stream.height
            metadata['fps'] = float(video_stream.average_rate) if video_stream.average_rate else 0
            metadata['frame_count'] = video_stream.frames
            metadata['duration'] = float(video_stream.duration * video_stream.time_base) if video_stream.duration and video_stream.time_base else 0
            container.close()
        # Add file information
        file_path = Path(video_path)
        metadata['filename'] = file_path.name
        metadata['file_extension'] = file_path.suffix
        metadata['file_size'] = os.path.getsize(video_path)
        return metadata
    except Exception as e:
        logger.error(f"Error extracting video metadata: {str(e)}")
        return {
            'filename': os.path.basename(video_path),
            'error': str(e)
        }


def extract_frames_at_intervals(
    video_path: str, 
    sampling_rate: float = 0.46, 
    device: str = 'cpu'
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Extract frames from a video at precise time intervals.
    
    Args:
        video_path: Path to the video file
        sampling_rate: Time interval between frames in seconds (default: 0.46)
        device: Device to use for frame extraction ('cpu' or 'cuda')
        
    Returns:
        tuple: (list of frames as numpy arrays, list of corresponding timestamps)
    """
    try:
        # Use OpenCV for frame extraction
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video with OpenCV: {video_path}")
            return [], []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate frame indices at precise intervals
        timestamps = []
        frame_indices = []
        
        current_time = 0
        while current_time < duration:
            timestamps.append(current_time)
            # Convert time to frame index
            frame_idx = int(current_time * fps)
            if frame_idx >= total_frames:
                frame_idx = total_frames - 1
            frame_indices.append(frame_idx)
            current_time += sampling_rate
        
        # Extract the frames at the calculated indices
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                logger.warning(f"Failed to read frame at index {idx}")
        
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames at {sampling_rate}s intervals")
        return frames, timestamps
        
    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
        return [], []


def preprocess_frames(
    frames: List[np.ndarray], 
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    to_rgb: bool = True
) -> np.ndarray:
    """
    Preprocess frames for model input.
    
    Args:
        frames: List of frames as numpy arrays
        target_size: Target size for resizing (height, width)
        normalize: Whether to normalize pixel values to [0,1]
        to_rgb: Whether to convert from BGR to RGB
        
    Returns:
        numpy.ndarray: Preprocessed frames
    """
    processed_frames = []
    
    for frame in frames:
        # Convert BGR to RGB if needed
        if to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame
        frame = cv2.resize(frame, (target_size[1], target_size[0]))
        
        # Normalize pixel values
        if normalize:
            frame = frame.astype(np.float32) / 255.0
        
        processed_frames.append(frame)
    
    return np.array(processed_frames)


def get_supported_video_formats() -> List[str]:
    """
    Get a list of supported video file extensions.
    
    Returns:
        list: List of supported video file extensions
    """
    return ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']


def is_video_file(file_path: str) -> bool:
    """
    Check if a file is a supported video file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bool: True if the file is a supported video file, False otherwise
    """
    ext = os.path.splitext(file_path)[1].lower()
    return ext in get_supported_video_formats()

