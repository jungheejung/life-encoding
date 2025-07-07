#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for video processing functionality.

This script tests the core video processing functions:
- Video validation
- Metadata extraction
- Frame extraction at precise intervals
- Frame preprocessing
"""

import os
import sys
import logging
import numpy as np
import cv2
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_video(output_path, duration=5, fps=30, width=640, height=480):
    """
    Create a test video file for testing.
    
    Args:
        output_path: Path to save the test video
        duration: Duration of the video in seconds
        fps: Frames per second
        width: Frame width
        height: Frame height
    """
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Generate frames
    total_frames = int(duration * fps)
    
    for i in range(total_frames):
        # Create a frame with a timestamp
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a timestamp
        timestamp = i / fps
        cv2.putText(
            frame, 
            f"Time: {timestamp:.2f}s", 
            (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2
        )
        
        # Add a frame counter
        cv2.putText(
            frame, 
            f"Frame: {i}", 
            (50, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2
        )
        
        # Add a colored rectangle that changes with time
        color = (
            int(255 * (i / total_frames)),
            int(255 * (1 - i / total_frames)),
            int(127 + 128 * np.sin(i / total_frames * np.pi))
        )
        cv2.rectangle(frame, (width // 4, height // 4), (3 * width // 4, 3 * height // 4), color, -1)
        
        # Write the frame
        out.write(frame)
    
    # Release the VideoWriter
    out.release()
    logger.info(f"Created test video at {output_path} ({duration}s, {fps} fps)")


def test_video_validation():
    """Test video validation functionality."""
    from video_annotation_system.utils.video_utils import validate_input
    
    logger.info("Testing video validation...")
    
    # Create a test video
    test_video_path = "test_video.mp4"
    if not os.path.exists(test_video_path):
        create_test_video(test_video_path, duration=2, fps=30)
    
    # Test valid video
    is_valid = validate_input(test_video_path)
    assert is_valid, "Video validation failed for valid video"
    
    # Test invalid video
    invalid_path = "nonexistent_video.mp4"
    is_valid = validate_input(invalid_path)
    assert not is_valid, "Video validation passed for invalid video"
    
    logger.info("Video validation tests passed")


def test_metadata_extraction():
    """Test metadata extraction functionality."""
    from video_annotation_system.utils.video_utils import get_video_metadata
    
    # Create a test video
    test_video_path = "test_video.mp4"
    if not os.path.exists(test_video_path):
        create_test_video(test_video_path)
    
    # Extract metadata
    metadata = get_video_metadata(test_video_path)
    
    # Check metadata
    expected_keys = ["width", "height", "fps", "frame_count", "duration", "filename"]
    has_all_keys = all(key in metadata for key in expected_keys)
    
    logger.info(f"Metadata: {metadata}")
    logger.info(f"Metadata extraction test: {'PASSED' if has_all_keys else 'FAILED'}")
    
    assert has_all_keys, "Metadata extraction test failed: missing expected keys"


def test_frame_extraction():
    """Test frame extraction at precise intervals."""
    from video_annotation_system.utils.video_utils import extract_frames_at_intervals
    
    # Create a test video
    test_video_path = "test_video.mp4"
    if not os.path.exists(test_video_path):
        create_test_video(test_video_path, duration=5, fps=30)
    
    # Extract frames at 0.46-second intervals
    frames, timestamps = extract_frames_at_intervals(test_video_path, sampling_rate=0.46)
    
    # Check frame count
    expected_frame_count = int(5 / 0.46) + 1  # 5 seconds at 0.46-second intervals
    actual_frame_count = len(frames)
    
    logger.info(f"Expected frame count: {expected_frame_count}")
    logger.info(f"Actual frame count: {actual_frame_count}")
    logger.info(f"Timestamps: {timestamps}")
    
    # Check timestamp intervals
    if len(timestamps) > 1:
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        avg_interval = sum(intervals) / len(intervals)
        max_deviation = max(abs(interval - 0.46) for interval in intervals)
        
        logger.info(f"Average interval: {avg_interval:.4f}s")
        logger.info(f"Maximum deviation: {max_deviation:.4f}s")
        
        interval_test_passed = abs(avg_interval - 0.46) < 0.01 and max_deviation < 0.01
        logger.info(f"Interval test: {'PASSED' if interval_test_passed else 'FAILED'}")
    else:
        interval_test_passed = False
        logger.info("Interval test: FAILED (not enough timestamps)")
    
    # Save a few frames for visual inspection
    if frames is not None and len(frames) > 0:
        os.makedirs("test_frames", exist_ok=True)
        for i, (frame, timestamp) in enumerate(zip(frames[:5], timestamps[:5])):
            output_path = f"test_frames/frame_{i}_{timestamp:.2f}s.png"
            cv2.imwrite(output_path, frame)
            logger.info(f"Saved frame {i} at {timestamp:.2f}s to {output_path}")
    
    frame_count_test_passed = abs(actual_frame_count - expected_frame_count) <= 1
    logger.info(f"Frame count test: {'PASSED' if frame_count_test_passed else 'FAILED'}")
    
    assert frame_count_test_passed, "Frame count test failed"
    assert interval_test_passed, "Interval test failed"


def test_frame_preprocessing():
    """Test frame preprocessing functionality."""
    from video_annotation_system.utils.video_utils import extract_frames_at_intervals, preprocess_frames
    
    logger.info("Testing frame preprocessing...")
    
    # Create a test video
    test_video_path = "test_video.mp4"
    if not os.path.exists(test_video_path):
        create_test_video(test_video_path, duration=2, fps=30)
    
    # Extract frames
    frames, timestamps = extract_frames_at_intervals(test_video_path, sampling_rate=0.46)
    assert frames is not None and timestamps is not None, "Failed to extract frames for preprocessing test"
    
    # Preprocess frames
    processed_frames = preprocess_frames(frames)
    assert processed_frames is not None, "Failed to preprocess frames"
    
    # Check shape
    expected_shape = (len(frames), 224, 224, 3)
    assert processed_frames.shape == expected_shape, f"Unexpected shape: {processed_frames.shape}, expected: {expected_shape}"
    
    logger.info("Frame preprocessing tests passed")


def main():
    """Run all tests."""
    logger.info("Starting video processing tests")
    
    tests = [
        ("Video validation", test_video_validation),
        ("Metadata extraction", test_metadata_extraction),
        ("Frame extraction", test_frame_extraction),
        ("Frame preprocessing", test_frame_preprocessing)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test")
        try:
            result = test_func()
            logger.info(f"{test_name} test: {'PASSED' if result else 'FAILED'}")
            all_passed = all_passed and result
        except Exception as e:
            logger.error(f"Error in {test_name} test: {str(e)}")
            all_passed = False
    
    logger.info(f"\nAll tests: {'PASSED' if all_passed else 'FAILED'}")
    
    # Clean up
    if os.path.exists("test_video.mp4"):
        os.remove("test_video.mp4")
        logger.info("Cleaned up test video")


if __name__ == "__main__":
    main()

