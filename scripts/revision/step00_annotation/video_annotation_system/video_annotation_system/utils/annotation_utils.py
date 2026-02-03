#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annotation Utilities Module

This module provides functions for handling annotations, including:
- Formatting annotations in the required structure
- Saving annotations in different formats (JSON, CSV, pickle)
- Validation of annotation completeness and consistency
"""

import os
import json
import csv
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_annotations(
    timestamps: List[float],
    action_annotations: List[List[str]],
    agent_annotations: List[List[str]],
    scene_annotations: List[List[str]]
) -> List[List]:
    """
    Format annotations into the required structure.
    
    Args:
        timestamps: List of timestamps
        action_annotations: List of action annotations for each timestamp
        agent_annotations: List of agent annotations for each timestamp
        scene_annotations: List of scene annotations for each timestamp
        
    Returns:
        list: Formatted annotations as nested lists
    """
    if not (len(timestamps) == len(action_annotations) == len(agent_annotations) == len(scene_annotations)):
        logger.error("Mismatch in annotation lengths")
        return []
    
    formatted_annotations = []
    
    for i, timestamp in enumerate(timestamps):
        # Ensure we have at least one annotation in each category
        actions = action_annotations[i] if action_annotations[i] else ["none"]
        agents = agent_annotations[i] if agent_annotations[i] else ["none"]
        scenes = scene_annotations[i] if scene_annotations[i] else ["none"]
        
        # Format as [timestamp, [actions], [agents], [backgrounds]]
        formatted_annotations.append([
            round(timestamp, 2),
            actions,
            agents,
            scenes
        ])
    
    return formatted_annotations


def save_annotations(
    annotations: List[List],
    video_path: str,
    metadata: Dict,
    output_dir: str = "./annotations",
    format: str = "json"
) -> str:
    """
    Save annotations to a file.
    
    Args:
        annotations: Formatted annotations
        video_path: Path to the source video
        metadata: Video metadata
        output_dir: Directory to save the annotations
        format: Output format (json, csv, or pickle)
        
    Returns:
        str: Path to the saved annotation file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]
    timestamp = metadata.get('timestamp', '')
    duration = metadata.get('duration', 0)
    
    output_filename = f"video_annotations_{video_name}_{duration:.2f}s.{format}"
    output_path = os.path.join(output_dir, output_filename)
    
    # Prepare data with metadata
    output_data = {
        "metadata": {
            "video_filename": video_filename,
            "duration": duration,
            "fps": metadata.get('fps', 0),
            "frame_count": metadata.get('frame_count', 0),
            "width": metadata.get('width', 0),
            "height": metadata.get('height', 0),
            "sampling_rate": 0.46,
            "annotation_count": len(annotations)
        },
        "annotations": annotations
    }
    
    # Save in the specified format
    try:
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
                
        elif format == "csv":
            # Flatten the nested structure for CSV
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(["timestamp", "actions", "agents", "backgrounds"])
                # Write data
                for ann in annotations:
                    writer.writerow([
                        ann[0],
                        "|".join(ann[1]),
                        "|".join(ann[2]),
                        "|".join(ann[3])
                    ])
                
        elif format == "pickle":
            with open(output_path, 'wb') as f:
                pickle.dump(output_data, f)
                
        else:
            logger.error(f"Unsupported format: {format}")
            return ""
        
        logger.info(f"Saved annotations to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving annotations: {str(e)}")
        return ""


def validate_annotations(annotations: List[List]) -> Tuple[bool, str]:
    """
    Validate annotations for completeness and consistency.
    
    Args:
        annotations: Formatted annotations
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not annotations:
        return False, "Empty annotations"
    
    # Check structure
    for i, ann in enumerate(annotations):
        if len(ann) != 4:
            return False, f"Invalid annotation structure at index {i}"
        
        # Check timestamp
        if not isinstance(ann[0], (int, float)):
            return False, f"Invalid timestamp at index {i}"
        
        # Check categories
        for j, category in enumerate(ann[1:], 1):
            if not isinstance(category, list):
                return False, f"Invalid category type at index {i}, category {j}"
            
            if not category:
                return False, f"Empty category at index {i}, category {j}"
    
    # Check temporal consistency
    timestamps = [ann[0] for ann in annotations]
    if not all(timestamps[i] < timestamps[i+1] for i in range(len(timestamps)-1)):
        return False, "Timestamps are not in ascending order"
    
    # Check sampling rate consistency
    if len(timestamps) > 1:
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        avg_interval = sum(intervals) / len(intervals)
        if not all(abs(interval - 0.46) < 0.01 for interval in intervals):
            return False, f"Inconsistent sampling rate (average: {avg_interval:.2f}s, expected: 0.46s)"
    
    return True, ""


def check_annotation_consistency(annotations: List[List]) -> Dict:
    """
    Check the consistency of annotations across adjacent timestamps.
    
    Args:
        annotations: Formatted annotations
        
    Returns:
        dict: Consistency metrics
    """
    if len(annotations) < 2:
        return {"error": "Not enough annotations to check consistency"}
    
    agent_consistency = []
    scene_consistency = []
    
    for i in range(len(annotations) - 1):
        # Check agent consistency
        prev_agents = set(annotations[i][2])
        curr_agents = set(annotations[i+1][2])
        if prev_agents and curr_agents:
            overlap = len(prev_agents.intersection(curr_agents)) / len(prev_agents.union(curr_agents))
            agent_consistency.append(overlap)
        
        # Check scene consistency
        prev_scenes = set(annotations[i][3])
        curr_scenes = set(annotations[i+1][3])
        if prev_scenes and curr_scenes:
            overlap = len(prev_scenes.intersection(curr_scenes)) / len(prev_scenes.union(curr_scenes))
            scene_consistency.append(overlap)
    
    return {
        "agent_consistency": sum(agent_consistency) / len(agent_consistency) if agent_consistency else 0,
        "scene_consistency": sum(scene_consistency) / len(scene_consistency) if scene_consistency else 0,
        "annotation_count": len(annotations)
    }

