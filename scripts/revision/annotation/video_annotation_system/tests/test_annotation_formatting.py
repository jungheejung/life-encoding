#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for annotation formatting and output functionality.

This script tests the annotation formatting and output functions:
- Annotation formatting
- Annotation saving in different formats (JSON, CSV, pickle)
- Annotation validation
- Annotation consistency checking
"""

import os
import sys
import logging
import json
import csv
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_annotation_formatting():
    """Test annotation formatting functionality."""
    from video_annotation_system.utils.annotation_utils import format_annotations
    
    logger.info("Testing annotation formatting...")
    
    # Create sample data
    timestamps = [0.0, 0.46, 0.92, 1.38, 1.84]
    action_annotations = [
        ["walking", "talking"],
        ["running"],
        ["jumping", "waving"],
        ["sitting", "typing"],
        ["standing"]
    ]
    agent_annotations = [
        ["person", "dog"],
        ["person"],
        ["person", "cat"],
        ["person"],
        ["person", "car"]
    ]
    scene_annotations = [
        ["outdoor", "park"],
        ["street"],
        ["beach"],
        ["office", "indoor"],
        ["parking lot"]
    ]
    
    # Format annotations
    formatted_annotations = format_annotations(
        timestamps, action_annotations, agent_annotations, scene_annotations
    )
    
    # Check formatted annotations
    assert formatted_annotations, "Failed to format annotations"
    assert len(formatted_annotations) == len(timestamps), f"Formatted annotation count mismatch: {len(formatted_annotations)} vs {len(timestamps)}"
    
    # Check annotation format
    for i, annotation in enumerate(formatted_annotations):
        assert isinstance(annotation, list) and len(annotation) == 4, f"Annotation at index {i} has invalid format"
        
        timestamp, actions, agents, scenes = annotation
        
        assert isinstance(timestamp, (int, float)), f"Timestamp at index {i} is not a number"
        assert isinstance(actions, list) and actions, f"Actions at index {i} is not a non-empty list"
        assert isinstance(agents, list) and agents, f"Agents at index {i} is not a non-empty list"
        assert isinstance(scenes, list) and scenes, f"Scenes at index {i} is not a non-empty list"
    
    logger.info(f"Formatted {len(formatted_annotations)} annotations")
    logger.info(f"Sample formatted annotations: {formatted_annotations[:2]}")


def test_annotation_saving_json():
    """Test saving annotations in JSON format."""
    from video_annotation_system.utils.annotation_utils import format_annotations, save_annotations
    
    logger.info("Testing annotation saving in JSON format...")
    
    # Create sample data
    timestamps = [0.0, 0.46, 0.92]
    action_annotations = [["walking"], ["running"], ["jumping"]]
    agent_annotations = [["person"], ["person", "dog"], ["person"]]
    scene_annotations = [["outdoor"], ["park"], ["street"]]
    
    # Format annotations
    formatted_annotations = format_annotations(
        timestamps, action_annotations, agent_annotations, scene_annotations
    )
    
    # Create metadata
    metadata = {
        "filename": "test_video.mp4",
        "duration": 5.0,
        "fps": 30.0,
        "frame_count": 150,
        "width": 640,
        "height": 480
    }
    
    # Create output directory
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save annotations in JSON format
    output_path = save_annotations(
        annotations=formatted_annotations,
        video_path="test_video.mp4",
        metadata=metadata,
        output_dir=output_dir,
        format="json"
    )
    
    # Check output path
    assert output_path, "Failed to save annotations in JSON format"
    assert os.path.exists(output_path), f"Output file not found: {output_path}"
    
    # Load and check saved annotations
    with open(output_path, 'r') as f:
        saved_data = json.load(f)
    
    assert "metadata" in saved_data and "annotations" in saved_data, "Saved JSON data missing required keys"
    assert len(saved_data["annotations"]) == len(formatted_annotations), f"Saved annotation count mismatch: {len(saved_data['annotations'])} vs {len(formatted_annotations)}"
    
    logger.info(f"Saved annotations to {output_path}")
    logger.info(f"Saved metadata: {saved_data['metadata']}")
    logger.info(f"Saved {len(saved_data['annotations'])} annotations in JSON format")


def test_annotation_saving_csv():
    """Test saving annotations in CSV format."""
    from video_annotation_system.utils.annotation_utils import format_annotations, save_annotations
    
    logger.info("Testing annotation saving in CSV format...")
    
    # Create sample data
    timestamps = [0.0, 0.46, 0.92]
    action_annotations = [["walking"], ["running"], ["jumping"]]
    agent_annotations = [["person"], ["person", "dog"], ["person"]]
    scene_annotations = [["outdoor"], ["park"], ["street"]]
    
    # Format annotations
    formatted_annotations = format_annotations(
        timestamps, action_annotations, agent_annotations, scene_annotations
    )
    
    # Create metadata
    metadata = {
        "filename": "test_video.mp4",
        "duration": 5.0,
        "fps": 30.0,
        "frame_count": 150,
        "width": 640,
        "height": 480
    }
    
    # Create output directory
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save annotations in CSV format
    output_path = save_annotations(
        annotations=formatted_annotations,
        video_path="test_video.mp4",
        metadata=metadata,
        output_dir=output_dir,
        format="csv"
    )
    
    # Check output path
    assert output_path, "Failed to save annotations in CSV format"
    assert os.path.exists(output_path), f"Output file not found: {output_path}"
    
    # Load and check saved annotations
    with open(output_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        saved_annotations = list(reader)
    
    assert len(header) == 4, "CSV header has incorrect number of columns"
    assert len(saved_annotations) == len(formatted_annotations), f"Saved annotation count mismatch: {len(saved_annotations)} vs {len(formatted_annotations)}"
    
    logger.info(f"Saved annotations to {output_path}")
    logger.info(f"CSV header: {header}")
    logger.info(f"Saved {len(saved_annotations)} annotations in CSV format")


def test_annotation_saving_pickle():
    """Test saving annotations in pickle format."""
    from video_annotation_system.utils.annotation_utils import format_annotations, save_annotations
    
    logger.info("Testing annotation saving in pickle format...")
    
    # Create sample data
    timestamps = [0.0, 0.46, 0.92]
    action_annotations = [["walking"], ["running"], ["jumping"]]
    agent_annotations = [["person"], ["person", "dog"], ["person"]]
    scene_annotations = [["outdoor"], ["park"], ["street"]]
    
    # Format annotations
    formatted_annotations = format_annotations(
        timestamps, action_annotations, agent_annotations, scene_annotations
    )
    
    # Create metadata
    metadata = {
        "filename": "test_video.mp4",
        "duration": 5.0,
        "fps": 30.0,
        "frame_count": 150,
        "width": 640,
        "height": 480
    }
    
    # Create output directory
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save annotations in pickle format
    output_path = save_annotations(
        annotations=formatted_annotations,
        video_path="test_video.mp4",
        metadata=metadata,
        output_dir=output_dir,
        format="pickle"
    )
    
    # Check output path
    assert output_path, "Failed to save annotations in pickle format"
    assert os.path.exists(output_path), f"Output file not found: {output_path}"
    
    # Load and check saved annotations
    with open(output_path, 'rb') as f:
        saved_data = pickle.load(f)
    
    assert "metadata" in saved_data and "annotations" in saved_data, "Saved pickle data missing required keys"
    assert len(saved_data["annotations"]) == len(formatted_annotations), f"Saved annotation count mismatch: {len(saved_data['annotations'])} vs {len(formatted_annotations)}"
    
    logger.info(f"Saved annotations to {output_path}")
    logger.info(f"Saved metadata: {saved_data['metadata']}")
    logger.info(f"Saved {len(saved_data['annotations'])} annotations in pickle format")


def test_annotation_validation():
    """Test annotation validation functionality."""
    from video_annotation_system.utils.annotation_utils import validate_annotations
    
    logger.info("Testing annotation validation...")
    
    # Create sample data
    annotations = {
        "timestamps": [0.0, 0.46, 0.92],
        "action_annotations": [["walking"], ["running"], ["jumping"]],
        "agent_annotations": [["person"], ["person", "dog"], ["person"]],
        "scene_annotations": [["outdoor"], ["park"], ["street"]]
    }
    
    # Test valid annotations
    is_valid = validate_annotations(annotations)
    assert is_valid, "Validation failed for valid annotations"
    
    # Test invalid annotations (mismatched lengths)
    invalid_annotations = {
        "timestamps": [0.0, 0.46],
        "action_annotations": [["walking"], ["running"], ["jumping"]],
        "agent_annotations": [["person"], ["person", "dog"]],
        "scene_annotations": [["outdoor"], ["park"], ["street"]]
    }
    
    is_valid = validate_annotations(invalid_annotations)
    assert not is_valid, "Validation passed for invalid annotations"
    
    logger.info("Annotation validation tests passed")


def test_annotation_consistency():
    """Test annotation consistency checking functionality."""
    from video_annotation_system.utils.annotation_utils import check_annotation_consistency
    
    logger.info("Testing annotation consistency...")
    
    # Create sample data
    annotations = {
        "timestamps": [0.0, 0.46, 0.92],
        "action_annotations": [["walking"], ["running"], ["jumping"]],
        "agent_annotations": [["person"], ["person", "dog"], ["person"]],
        "scene_annotations": [["outdoor"], ["park"], ["street"]]
    }
    
    # Test consistent annotations
    is_consistent = check_annotation_consistency(annotations)
    assert is_consistent, "Consistency check failed for consistent annotations"
    
    # Test inconsistent annotations (conflicting agent and action)
    inconsistent_annotations = {
        "timestamps": [0.0, 0.46, 0.92],
        "action_annotations": [["walking"], ["running"], ["jumping"]],
        "agent_annotations": [["person"], ["car"], ["person"]],  # Car can't run or jump
        "scene_annotations": [["outdoor"], ["park"], ["street"]]
    }
    
    is_consistent = check_annotation_consistency(inconsistent_annotations)
    assert not is_consistent, "Consistency check passed for inconsistent annotations"
    
    logger.info("Annotation consistency tests passed")


def main():
    """Run all tests."""
    logger.info("Starting annotation formatting and output tests")
    
    tests = [
        ("Annotation formatting", test_annotation_formatting),
        ("Annotation saving (JSON)", test_annotation_saving_json),
        ("Annotation saving (CSV)", test_annotation_saving_csv),
        ("Annotation saving (pickle)", test_annotation_saving_pickle),
        ("Annotation validation", test_annotation_validation),
        ("Annotation consistency", test_annotation_consistency)
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
    if os.path.exists("test_output"):
        import shutil
        shutil.rmtree("test_output")
        logger.info("Cleaned up test output directory")


if __name__ == "__main__":
    main()

