#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic test script to verify the project structure and imports.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test importing the project modules."""
    from video_annotation_system.utils import video_utils, annotation_utils
    from video_annotation_system.models import model_loader, annotation_generator
    
    logger.info("Successfully imported all modules")


def test_video_utils():
    """Test basic functionality of the video_utils module."""
    from video_annotation_system.utils.video_utils import get_supported_video_formats, is_video_file
    
    formats = get_supported_video_formats()
    logger.info(f"Supported video formats: {formats}")
    assert formats, "No supported video formats found"
    
    test_file = "test.mp4"
    is_video = is_video_file(test_file)
    logger.info(f"Is {test_file} a video file? {is_video}")
    assert isinstance(is_video, bool), "is_video_file did not return a boolean"


def test_annotation_utils():
    """Test basic functionality of the annotation_utils module."""
    from video_annotation_system.utils.annotation_utils import format_annotations
    
    # Create sample data
    timestamps = [0.0, 0.46, 0.92]
    action_annotations = [["walking"], ["running"], ["jumping"]]
    agent_annotations = [["person"], ["person", "dog"], ["person"]]
    scene_annotations = [["outdoor"], ["park"], ["street"]]
    
    # Format annotations
    formatted = format_annotations(timestamps, action_annotations, agent_annotations, scene_annotations)
    logger.info(f"Formatted annotations: {formatted}")
    
    assert formatted, "Failed to format annotations"
    assert len(formatted) == len(timestamps), "Formatted annotation count mismatch"
    
    # Check annotation format
    for i, annotation in enumerate(formatted):
        assert isinstance(annotation, list) and len(annotation) == 4, f"Annotation at index {i} has invalid format"
        timestamp, actions, agents, scenes = annotation
        assert isinstance(timestamp, (int, float)), f"Timestamp at index {i} is not a number"
        assert isinstance(actions, list) and actions, f"Actions at index {i} is not a non-empty list"
        assert isinstance(agents, list) and agents, f"Agents at index {i} is not a non-empty list"
        assert isinstance(scenes, list) and scenes, f"Scenes at index {i} is not a non-empty list"


def main():
    """Run all tests."""
    logger.info("Starting basic tests")
    
    tests = [
        ("Import test", test_imports),
        ("Video utils test", test_video_utils),
        ("Annotation utils test", test_annotation_utils)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        logger.info(f"Running {test_name}")
        result = test_func()
        logger.info(f"{test_name}: {'PASSED' if result else 'FAILED'}")
        all_passed = all_passed and result
    
    logger.info(f"All tests: {'PASSED' if all_passed else 'FAILED'}")


if __name__ == "__main__":
    main()

