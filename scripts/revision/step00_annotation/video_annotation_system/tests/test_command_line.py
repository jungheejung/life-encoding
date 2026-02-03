#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for command-line interface and batch processing functionality.

This script tests the command-line interface and batch processing functions:
- Command-line argument parsing
- Single video processing
- Batch folder processing
- Output directory structure maintenance
- Configuration options for models and processing
"""

import os
import sys
import logging
import subprocess
import tempfile
import shutil
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_video(output_path, duration=2, fps=30, width=320, height=240):
    """Create a test video file for testing."""
    import cv2
    import numpy as np
    
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
            0.5, 
            (255, 255, 255), 
            1
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


def setup_test_environment():
    """Set up a test environment with videos and directories."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory: {temp_dir}")
    
    # Create subdirectories
    input_dir = os.path.join(temp_dir, "input")
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a test video in the input directory
    video_path = os.path.join(input_dir, "test_video.mp4")
    create_test_video(video_path)
    
    # Create a subdirectory with more test videos
    subdir = os.path.join(input_dir, "subdir")
    os.makedirs(subdir, exist_ok=True)
    
    # Create test videos in the subdirectory
    for i in range(2):
        video_path = os.path.join(subdir, f"test_video_{i}.mp4")
        create_test_video(video_path, duration=1 + i)
    
    return temp_dir, input_dir, output_dir


def cleanup_test_environment(temp_dir):
    """Clean up the test environment."""
    shutil.rmtree(temp_dir)
    logger.info(f"Cleaned up temporary directory: {temp_dir}")


def test_command_line_help():
    """Test the command-line help message."""
    logger.info("Testing command-line help message...")
    
    # Run the script with --help
    result = subprocess.run(
        ["python", "video_annotator.py", "--help"],
        capture_output=True,
        text=True,
        check=True
    )
    
    # Check if the help message contains expected sections
    help_text = result.stdout
    
    expected_sections = [
        "usage:",
        "--input",
        "--input-folder",
        "--model",
        "--action-model",
        "--agent-model",
        "--scene-model",
        "--output-dir",
        "--format",
        "--sampling-rate",
        "--batch-size",
        "--device"
    ]
    
    missing_sections = [section for section in expected_sections if section not in help_text]
    assert not missing_sections, f"Help message missing expected sections: {missing_sections}"
    
    logger.info("Help message contains all expected sections")


def _test_single_video_processing(input_dir, output_dir):
    """Test processing a single video."""
    logger.info("Testing single video processing...")
    
    # Get the path to a test video
    video_path = os.path.join(input_dir, "test_video.mp4")
    assert os.path.exists(video_path), f"Test video not found: {video_path}"
    
    # Run the script with the test video
    command = [
        "python", "video_annotator.py",
        "--input", video_path,
        "--output-dir", output_dir,
        "--format", "json",
        "--sampling-rate", "0.46",
        "--batch-size", "2"
    ]
    
    logger.info(f"Running command: {' '.join(command)}")
    
    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )
    
    # Check if the command was successful
    assert result.returncode == 0, f"Command failed with return code {result.returncode}\nStdout: {result.stdout}\nStderr: {result.stderr}"
    
    # Check if output file was created
    output_files = os.listdir(output_dir)
    assert output_files, f"No output files found in {output_dir}"
    
    # Check if output file has the expected name pattern
    expected_pattern = "video_annotations_test_video"
    matching_files = [f for f in output_files if expected_pattern in f]
    assert matching_files, f"No output files matching pattern '{expected_pattern}' found in {output_dir}"
    
    # Check if output file is a valid JSON file
    output_file = os.path.join(output_dir, matching_files[0])
    with open(output_file, 'r') as f:
        import json
        data = json.load(f)
    
    assert "metadata" in data, "Output file missing metadata"
    assert "annotations" in data, "Output file missing annotations"
    assert isinstance(data["annotations"], list), "Annotations is not a list"
    assert len(data["annotations"]) > 0, "No annotations found in output file"
    
    logger.info(f"Successfully processed video and created output file: {output_file}")


def _test_batch_folder_processing(input_dir, output_dir):
    """Test processing a folder of videos."""
    logger.info("Testing batch folder processing...")
    
    # Run the script with the input folder
    command = [
        "python", "video_annotator.py",
        "--input-folder", input_dir,
        "--output-dir", output_dir,
        "--format", "json",
        "--sampling-rate", "0.46",
        "--batch-size", "2"
    ]
    
    logger.info(f"Running command: {' '.join(command)}")
    
    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )
    
    # Check if the command was successful
    assert result.returncode == 0, f"Command failed with return code {result.returncode}\nStdout: {result.stdout}\nStderr: {result.stderr}"
    
    # Check if output files were created
    output_files = os.listdir(output_dir)
    assert output_files, f"No output files found in {output_dir}"
    
    # Check if output files have the expected name patterns
    expected_patterns = [
        "video_annotations_test_video",
        "video_annotations_test_video_0",
        "video_annotations_test_video_1"
    ]
    
    for pattern in expected_patterns:
        matching_files = [f for f in output_files if pattern in f]
        assert matching_files, f"No output files matching pattern '{pattern}' found in {output_dir}"
    
    # Check if output files are valid JSON files
    for pattern in expected_patterns:
        matching_files = [f for f in output_files if pattern in f]
        output_file = os.path.join(output_dir, matching_files[0])
        
        with open(output_file, 'r') as f:
            import json
            data = json.load(f)
        
        assert "metadata" in data, f"Output file {output_file} missing metadata"
        assert "annotations" in data, f"Output file {output_file} missing annotations"
        assert isinstance(data["annotations"], list), f"Annotations in {output_file} is not a list"
        assert len(data["annotations"]) > 0, f"No annotations found in output file {output_file}"
    
    logger.info(f"Successfully processed folder and created {len(output_files)} output files")


def _test_model_configuration(input_dir, output_dir):
    """Test different model configurations."""
    logger.info("Testing model configurations...")
    
    # Get the path to a test video
    video_path = os.path.join(input_dir, "test_video.mp4")
    assert os.path.exists(video_path), f"Test video not found: {video_path}"
    
    # Test different model combinations
    model_configs = [
        ["--model", "facebook/slowfast"],
        ["--action-model", "facebook/slowfast", "--agent-model", "openai/clip-vit-base-patch32"],
        ["--action-model", "facebook/slowfast", "--scene-model", "google/vit-base-patch16-224"],
        ["--agent-model", "openai/clip-vit-base-patch32", "--scene-model", "google/vit-base-patch16-224"],
        ["--action-model", "facebook/slowfast", "--agent-model", "openai/clip-vit-base-patch32", "--scene-model", "google/vit-base-patch16-224"]
    ]
    
    for config in model_configs:
        # Run the script with the current model configuration
        command = [
            "python", "video_annotator.py",
            "--input", video_path,
            "--output-dir", output_dir,
            "--format", "json",
            "--sampling-rate", "0.46",
            "--batch-size", "2"
        ] + config
        
        logger.info(f"Running command: {' '.join(command)}")
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )
        
        # Check if the command was successful
        assert result.returncode == 0, f"Command failed with return code {result.returncode}\nStdout: {result.stdout}\nStderr: {result.stderr}"
        
        # Check if output file was created
        output_files = os.listdir(output_dir)
        assert output_files, f"No output files found in {output_dir}"
        
        # Check if output file is a valid JSON file
        output_file = os.path.join(output_dir, output_files[-1])  # Get the most recent output file
        with open(output_file, 'r') as f:
            import json
            data = json.load(f)
        
        assert "metadata" in data, f"Output file {output_file} missing metadata"
        assert "annotations" in data, f"Output file {output_file} missing annotations"
        assert isinstance(data["annotations"], list), f"Annotations in {output_file} is not a list"
        assert len(data["annotations"]) > 0, f"No annotations found in output file {output_file}"
        
        logger.info(f"Successfully processed video with model configuration: {' '.join(config)}")


def main():
    """Run all tests."""
    logger.info("Starting command-line interface and batch processing tests")
    
    # Test command-line help
    help_test_result = test_command_line_help()
    logger.info(f"Command-line help test: {'PASSED' if help_test_result else 'FAILED'}")
    
    # Set up test environment
    temp_dir, input_dir, output_dir = setup_test_environment()
    
    try:
        # Run tests that require the test environment
        tests = [
            ("Single video processing", lambda: _test_single_video_processing(input_dir, output_dir)),
            ("Batch folder processing", lambda: _test_batch_folder_processing(input_dir, output_dir)),
            ("Model configuration", lambda: _test_model_configuration(input_dir, output_dir))
        ]
        
        all_passed = help_test_result
        
        for test_name, test_func in tests:
            logger.info(f"\nRunning {test_name} test")
            try:
                # Clear output directory before each test
                for file in os.listdir(output_dir):
                    os.remove(os.path.join(output_dir, file))
                
                result = test_func()
                logger.info(f"{test_name} test: {'PASSED' if result else 'FAILED'}")
                all_passed = all_passed and result
            except Exception as e:
                logger.error(f"Error in {test_name} test: {str(e)}")
                all_passed = False
        
        logger.info(f"\nAll tests: {'PASSED' if all_passed else 'FAILED'}")
    
    finally:
        # Clean up test environment
        cleanup_test_environment(temp_dir)


if __name__ == "__main__":
    main()

