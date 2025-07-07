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
    scene_annotations: List[List[str]],
    transcript_annotations: Optional[List[List[str]]] = None
) -> List[List]:
    """
    Format annotations into the required structure.
    
    Args:
        timestamps: List of timestamps
        action_annotations: List of action annotations for each timestamp
        agent_annotations: List of agent annotations for each timestamp
        scene_annotations: List of scene annotations for each timestamp
        transcript_annotations: List of transcript annotations for each timestamp (optional)
        
    Returns:
        list: Formatted annotations as nested lists
    """
    # Check if all annotation lists have the same length
    annotation_lengths = [len(timestamps), len(action_annotations), len(agent_annotations), len(scene_annotations)]
    if transcript_annotations:
        annotation_lengths.append(len(transcript_annotations))
    
    if len(set(annotation_lengths)) != 1:
        logger.error(f"Mismatch in annotation lengths: {annotation_lengths}")
        return []
    
    formatted_annotations = []
    
    for i, timestamp in enumerate(timestamps):
        # Ensure we have at least one annotation in each category
        actions = action_annotations[i] if action_annotations[i] else ["none"]
        agents = agent_annotations[i] if agent_annotations[i] else ["none"]
        scenes = scene_annotations[i] if scene_annotations[i] else ["none"]
        
        # Format as [timestamp, [actions], [agents], [backgrounds], [transcript]]
        annotation = [
            round(timestamp, 2),
            actions,
            agents,
            scenes
        ]
        
        # Add transcript if available
        if transcript_annotations:
            transcript = transcript_annotations[i] if transcript_annotations[i] else ["none"]
            annotation.append(transcript)
        
        formatted_annotations.append(annotation)
    
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
    
    # Determine if annotations include transcript
    has_transcript = len(annotations[0]) > 4 if annotations else False
    
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
            "annotation_count": len(annotations),
            "has_transcript": has_transcript
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
                header = ["timestamp", "actions", "agents", "backgrounds"]
                if has_transcript:
                    header.append("transcript")
                writer.writerow(header)
                
                # Write data
                for ann in annotations:
                    row = [
                        ann[0],
                        "|".join(ann[1]),
                        "|".join(ann[2]),
                        "|".join(ann[3])
                    ]
                    if has_transcript and len(ann) > 4:
                        row.append("|".join(ann[4]))
                    writer.writerow(row)
                
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
    expected_length = len(annotations[0])  # Use first annotation as reference
    
    for i, ann in enumerate(annotations):
        if len(ann) != expected_length:
            return False, f"Inconsistent annotation structure at index {i}"
        
        # Check timestamp
        if not isinstance(ann[0], (int, float)):
            return False, f"Invalid timestamp at index {i}"
        
        # Check categories (actions, agents, backgrounds, and optionally transcript)
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
    if not annotations or len(annotations) < 2:
        return {"error": "Not enough annotations to check consistency"}
    
    # Initialize counters
    action_changes = 0
    agent_changes = 0
    scene_changes = 0
    transcript_changes = 0
    
    # Count changes between adjacent annotations
    for i in range(len(annotations) - 1):
        # Actions
        if set(annotations[i][1]) != set(annotations[i+1][1]):
            action_changes += 1
        
        # Agents
        if set(annotations[i][2]) != set(annotations[i+1][2]):
            agent_changes += 1
        
        # Scenes
        if set(annotations[i][3]) != set(annotations[i+1][3]):
            scene_changes += 1
        
        # Transcript (if available)
        if len(annotations[i]) > 4 and len(annotations[i+1]) > 4:
            if set(annotations[i][4]) != set(annotations[i+1][4]):
                transcript_changes += 1
    
    # Calculate change rates
    total_transitions = len(annotations) - 1
    metrics = {
        "action_change_rate": action_changes / total_transitions,
        "agent_change_rate": agent_changes / total_transitions,
        "scene_change_rate": scene_changes / total_transitions,
    }
    
    # Add transcript metrics if available
    if len(annotations[0]) > 4:
        metrics["transcript_change_rate"] = transcript_changes / total_transitions
    
    return metrics


def extract_annotation_statistics(annotations: List[List]) -> Dict:
    """
    Extract statistics from annotations.
    
    Args:
        annotations: Formatted annotations
        
    Returns:
        dict: Annotation statistics
    """
    if not annotations:
        return {"error": "Empty annotations"}
    
    # Initialize counters
    unique_actions = set()
    unique_agents = set()
    unique_scenes = set()
    unique_transcripts = set()
    
    # Count unique annotations
    for ann in annotations:
        unique_actions.update(ann[1])
        unique_agents.update(ann[2])
        unique_scenes.update(ann[3])
        
        # Transcript (if available)
        if len(ann) > 4:
            unique_transcripts.update(ann[4])
    
    # Calculate statistics
    stats = {
        "annotation_count": len(annotations),
        "unique_actions": len(unique_actions),
        "unique_agents": len(unique_agents),
        "unique_scenes": len(unique_scenes),
        "most_common_actions": get_most_common(annotations, 1),
        "most_common_agents": get_most_common(annotations, 2),
        "most_common_scenes": get_most_common(annotations, 3)
    }
    
    # Add transcript statistics if available
    if len(annotations[0]) > 4:
        stats["unique_transcripts"] = len(unique_transcripts)
        stats["most_common_transcripts"] = get_most_common(annotations, 4)
    
    return stats


def get_most_common(annotations: List[List], category_index: int, top_n: int = 5) -> List[Tuple[str, int]]:
    """
    Get the most common annotations in a category.
    
    Args:
        annotations: Formatted annotations
        category_index: Index of the category in the annotation list
        top_n: Number of top items to return
        
    Returns:
        list: List of (annotation, count) tuples
    """
    if not annotations or category_index >= len(annotations[0]):
        return []
    
    # Count occurrences
    counts = {}
    for ann in annotations:
        for item in ann[category_index]:
            counts[item] = counts.get(item, 0) + 1
    
    # Sort by count (descending)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N
    return sorted_counts[:top_n]

