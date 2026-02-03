#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for annotation generation functionality.

This script tests the annotation generation functions:
- Action annotation generation
- Agent annotation generation
- Scene annotation generation
- Combined annotation generation
"""

import os
import sys
import logging
import numpy as np
import cv2
import torch
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test annotation generation functionality")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to use for model inference")
    return parser.parse_args()


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


def create_mock_args(device="cpu"):
    """Create mock command-line arguments for testing."""
    class MockArgs:
        def __init__(self, device):
            self.device = device
            self.model = None
            self.action_model = "facebook/slowfast"
            self.agent_model = "openai/clip-vit-base-patch32"
            self.scene_model = "google/vit-base-patch16-224"
            self.batch_size = 2
            self.sampling_rate = 0.46
    
    return MockArgs(device)


def test_action_annotation_generation(device="cpu"):
    """Test action annotation generation."""
    from video_annotation_system.utils.video_utils import extract_frames_at_intervals
    from video_annotation_system.models.model_loader import load_slowfast_model
    from video_annotation_system.models.annotation_generator import generate_action_annotations
    
    logger.info("Testing action annotation generation...")
    
    # Create a test video
    test_video_path = "test_video.mp4"
    if not os.path.exists(test_video_path):
        create_test_video(test_video_path, duration=2, fps=30)
    
    # Extract frames
    frames, timestamps = extract_frames_at_intervals(test_video_path, sampling_rate=0.46)
    
    assert frames and timestamps, "Failed to extract frames for action annotation test"
    
    # Load action model
    model_dict = load_slowfast_model("facebook/slowfast", device)
    assert model_dict, "Failed to load action model"
    
    # Generate action annotations
    action_annotations = generate_action_annotations(
        frames, timestamps, model_dict, batch_size=2, device=device
    )
    
    # Check annotations
    assert action_annotations, "Failed to generate action annotations"
    assert len(action_annotations) == len(timestamps), f"Action annotation count mismatch: {len(action_annotations)} vs {len(timestamps)}"
    
    # Check annotation format
    for i, annotation in enumerate(action_annotations):
        assert isinstance(annotation, list), f"Action annotation at index {i} is not a list"
        assert annotation, f"Empty action annotation at index {i}"
        assert all(isinstance(action, str) for action in annotation), f"Action annotation at index {i} contains non-string elements"
    
    logger.info(f"Generated {len(action_annotations)} action annotations")
    logger.info(f"Sample action annotations: {action_annotations[:3]}")


def test_agent_annotation_generation(device="cpu"):
    """Test agent annotation generation."""
    from video_annotation_system.utils.video_utils import extract_frames_at_intervals
    from video_annotation_system.models.model_loader import load_clip_model
    from video_annotation_system.models.annotation_generator import generate_agent_annotations
    
    logger.info("Testing agent annotation generation...")
    
    # Create a test video
    test_video_path = "test_video.mp4"
    if not os.path.exists(test_video_path):
        create_test_video(test_video_path, duration=2, fps=30)
    
    # Extract frames
    frames, timestamps = extract_frames_at_intervals(test_video_path, sampling_rate=0.46)
    
    assert frames and timestamps, "Failed to extract frames for agent annotation test"
    
    # Load agent model
    model_dict = load_clip_model("openai/clip-vit-base-patch32", device)
    assert model_dict, "Failed to load agent model"
    
    # Generate agent annotations
    agent_annotations = generate_agent_annotations(
        frames, timestamps, model_dict, batch_size=2, device=device
    )
    
    # Check annotations
    assert agent_annotations, "Failed to generate agent annotations"
    assert len(agent_annotations) == len(timestamps), f"Agent annotation count mismatch: {len(agent_annotations)} vs {len(timestamps)}"
    
    # Check annotation format
    for i, annotation in enumerate(agent_annotations):
        assert isinstance(annotation, list), f"Agent annotation at index {i} is not a list"
        assert annotation, f"Empty agent annotation at index {i}"
        assert all(isinstance(agent, str) for agent in annotation), f"Agent annotation at index {i} contains non-string elements"
    
    logger.info(f"Generated {len(agent_annotations)} agent annotations")
    logger.info(f"Sample agent annotations: {agent_annotations[:3]}")


def test_scene_annotation_generation(device="cpu"):
    """Test scene annotation generation."""
    from video_annotation_system.utils.video_utils import extract_frames_at_intervals
    from video_annotation_system.models.model_loader import load_vit_model
    from video_annotation_system.models.annotation_generator import generate_scene_annotations
    
    logger.info("Testing scene annotation generation...")
    
    # Create a test video
    test_video_path = "test_video.mp4"
    if not os.path.exists(test_video_path):
        create_test_video(test_video_path, duration=2, fps=30)
    
    # Extract frames
    frames, timestamps = extract_frames_at_intervals(test_video_path, sampling_rate=0.46)
    
    assert frames and timestamps, "Failed to extract frames for scene annotation test"
    
    # Load scene model
    model_dict = load_vit_model("google/vit-base-patch16-224", device)
    assert model_dict, "Failed to load scene model"
    
    # Generate scene annotations
    scene_annotations = generate_scene_annotations(
        frames, timestamps, model_dict, batch_size=2, device=device
    )
    
    # Check annotations
    assert scene_annotations, "Failed to generate scene annotations"
    assert len(scene_annotations) == len(timestamps), f"Scene annotation count mismatch: {len(scene_annotations)} vs {len(timestamps)}"
    
    # Check annotation format
    for i, annotation in enumerate(scene_annotations):
        assert isinstance(annotation, list), f"Scene annotation at index {i} is not a list"
        assert annotation, f"Empty scene annotation at index {i}"
        assert all(isinstance(scene, str) for scene in annotation), f"Scene annotation at index {i} contains non-string elements"
    
    logger.info(f"Generated {len(scene_annotations)} scene annotations")
    logger.info(f"Sample scene annotations: {scene_annotations[:3]}")


def test_combined_annotation_generation(device="cpu"):
    """Test combined annotation generation."""
    from video_annotation_system.utils.video_utils import extract_frames_at_intervals
    from video_annotation_system.models.model_loader import (
        load_slowfast_model,
        load_clip_model,
        load_vit_model
    )
    from video_annotation_system.models.annotation_generator import (
        generate_action_annotations,
        generate_agent_annotations,
        generate_scene_annotations
    )
    
    logger.info("Testing combined annotation generation...")
    
    # Create a test video
    test_video_path = "test_video.mp4"
    if not os.path.exists(test_video_path):
        create_test_video(test_video_path, duration=2, fps=30)
    
    # Extract frames
    frames, timestamps = extract_frames_at_intervals(test_video_path, sampling_rate=0.46)
    
    assert frames and timestamps, "Failed to extract frames for combined annotation test"
    
    # Load models
    action_model = load_slowfast_model("facebook/slowfast", device)
    agent_model = load_clip_model("openai/clip-vit-base-patch32", device)
    scene_model = load_vit_model("google/vit-base-patch16-224", device)
    
    assert action_model, "Failed to load action model"
    assert agent_model, "Failed to load agent model"
    assert scene_model, "Failed to load scene model"
    
    # Generate annotations
    action_annotations = generate_action_annotations(
        frames, timestamps, action_model, batch_size=2, device=device
    )
    agent_annotations = generate_agent_annotations(
        frames, timestamps, agent_model, batch_size=2, device=device
    )
    scene_annotations = generate_scene_annotations(
        frames, timestamps, scene_model, batch_size=2, device=device
    )
    
    # Check annotations
    assert action_annotations, "Failed to generate action annotations"
    assert agent_annotations, "Failed to generate agent annotations"
    assert scene_annotations, "Failed to generate scene annotations"
    
    assert len(action_annotations) == len(timestamps), f"Action annotation count mismatch: {len(action_annotations)} vs {len(timestamps)}"
    assert len(agent_annotations) == len(timestamps), f"Agent annotation count mismatch: {len(agent_annotations)} vs {len(timestamps)}"
    assert len(scene_annotations) == len(timestamps), f"Scene annotation count mismatch: {len(scene_annotations)} vs {len(timestamps)}"
    
    # Check annotation format
    for i in range(len(timestamps)):
        assert isinstance(action_annotations[i], list), f"Action annotation at index {i} is not a list"
        assert isinstance(agent_annotations[i], list), f"Agent annotation at index {i} is not a list"
        assert isinstance(scene_annotations[i], list), f"Scene annotation at index {i} is not a list"
        
        assert action_annotations[i], f"Empty action annotation at index {i}"
        assert agent_annotations[i], f"Empty agent annotation at index {i}"
        assert scene_annotations[i], f"Empty scene annotation at index {i}"
        
        assert all(isinstance(action, str) for action in action_annotations[i]), f"Action annotation at index {i} contains non-string elements"
        assert all(isinstance(agent, str) for agent in agent_annotations[i]), f"Agent annotation at index {i} contains non-string elements"
        assert all(isinstance(scene, str) for scene in scene_annotations[i]), f"Scene annotation at index {i} contains non-string elements"
    
    logger.info(f"Generated {len(timestamps)} combined annotations")
    logger.info(f"Sample combined annotations:")
    for i in range(min(3, len(timestamps))):
        logger.info(f"Timestamp {timestamps[i]:.2f}s:")
        logger.info(f"  Actions: {action_annotations[i]}")
        logger.info(f"  Agents: {agent_annotations[i]}")
        logger.info(f"  Scenes: {scene_annotations[i]}")


def main():
    """Run all tests."""
    args = parse_args()
    device = args.device
    
    logger.info(f"Starting annotation generation tests on device: {device}")
    
    # Check if CUDA is available when requested
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    tests = [
        ("Action annotation generation", lambda: test_action_annotation_generation(device)),
        ("Agent annotation generation", lambda: test_agent_annotation_generation(device)),
        ("Scene annotation generation", lambda: test_scene_annotation_generation(device)),
        ("Combined annotation generation", lambda: test_combined_annotation_generation(device))
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

