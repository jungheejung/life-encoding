#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annotation Generator Module

This module handles generating annotations from video frames using the loaded models:
- Action annotations using SlowFast or similar models
- Agent annotations using CLIP or similar models
- Scene/background annotations using ViT or similar models
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Union, Any
from tqdm import tqdm
import pandas as pd

from ..utils.video_utils import extract_frames_at_intervals, preprocess_frames

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_annotations(
    video_path: str,
    models: Dict,
    sampling_rate: float = 0.46,
    batch_size: int = 8,
    device: str = "cpu"
) -> List[List]:
    """
    Generate annotations for a video using the loaded models.
    
    Args:
        video_path: Path to the video file
        models: Dictionary of loaded models
        sampling_rate: Time interval between frames in seconds
        batch_size: Batch size for model inference
        device: Device to use for inference ('cpu' or 'cuda')
        
    Returns:
        list: Formatted annotations
    """
    # Extract frames at the specified sampling rate
    frames, timestamps = extract_frames_at_intervals(video_path, sampling_rate, device)
    
    if not frames or not timestamps:
        logger.error("Failed to extract frames from video")
        return []
    
    logger.info(f"Generating annotations for {len(frames)} frames")
    
    # Generate annotations for each category
    action_annotations = generate_action_annotations(frames, timestamps, models.get("action"), batch_size, device)
    agent_annotations = generate_agent_annotations(frames, timestamps, models.get("agent"), batch_size, device)
    scene_annotations = generate_scene_annotations(frames, timestamps, models.get("scene"), batch_size, device)
    object_annotations = generate_object_annotations(frames, timestamps, models.get("object"), batch_size, device)
    
    # Format annotations
    formatted_annotations = []
    for i, timestamp in enumerate(timestamps):
        formatted_annotations.append([
            round(timestamp, 2),
            action_annotations[i],
            agent_annotations[i],
            scene_annotations[i],
            object_annotations[i]
        ])
    
    return formatted_annotations


def generate_action_annotations(
    frames: List[np.ndarray],
    timestamps: List[float],
    model_dict: Dict,
    batch_size: int = 8,
    device: str = "cpu"
) -> List[List[str]]:
    """
    Generate action annotations for video frames.
    
    Args:
        frames: List of video frames
        timestamps: List of frame timestamps
        model_dict: Dictionary containing the action model and processor
        batch_size: Batch size for model inference
        device: Device to use for inference
        
    Returns:
        list: List of action annotations for each frame
    """
    if not model_dict:
        logger.error("Action model not provided")
        return [["unknown_action"] for _ in timestamps]
    
    model = model_dict["model"]
    processor = model_dict["processor"]
    model_type = model_dict["type"]
    
    action_annotations = []
    
    try:
        # Process frames in batches
        for i in tqdm(range(0, len(frames), batch_size), desc="Generating action annotations"):
            batch_frames = frames[i:i+batch_size]
            
            if model_type == "slowfast":
                # SlowFast-specific processing
                # This is a simplified implementation
                batch_inputs = processor(batch_frames)
                with torch.no_grad():
                    outputs = model(batch_inputs)
                
                # For demonstration, generate placeholder action annotations
                # In a real implementation, this would use the actual model outputs
                for _ in range(len(batch_frames)):
                    # Example actions
                    actions = pd.read_csv('/Users/h/Documents/projects_local/life-encoding/scripts/revision/word_corpus/action_words_gerunds_trim.txt', sep='\t', header=None)[0].tolist()
                    # actions = ["walking", "running", "sitting", "standing", "talking"]
                    # Randomly select 1-3 actions for demonstration
                    num_actions = np.random.randint(1, 4)
                    selected_actions = np.random.choice(actions, size=num_actions, replace=False)
                    action_annotations.append(selected_actions.tolist())
            
            elif model_type == "clip":
                # CLIP-specific processing for actions
                action_prompts = pd.read_csv('/Users/h/Documents/projects_local/life-encoding/scripts/revision/word_corpus/action_words_gerunds_trim.txt', sep='\t', header=None)[0]
                # action_prompts = [
                #     "walking", "running", "jumping", "sitting", "standing",
                #     "talking", "eating", "drinking", "sleeping", "dancing",
                #     "fighting", "climbing", "swimming", "driving", "riding",
                #     "playing", "working", "exercising", "cooking", "reading"
                # ]
                
                for frame in batch_frames:
                    # Process the frame with CLIP
                    inputs = processor(
                        text=action_prompts,
                        images=frame,
                        return_tensors="pt",
                        padding=True
                    ).to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        probs = logits_per_image.softmax(dim=1)
                    
                    # Get top 3 actions
                    top_probs, top_indices = probs.topk(3)
                    selected_actions = [action_prompts[idx] for idx in top_indices[0].tolist()]
                    action_annotations.append(selected_actions)
            
            elif model_type == "vit":
                # ViT-specific processing for actions
                # This is a simplified implementation
                for frame in batch_frames:
                    # Process the frame with ViT
                    inputs = processor(frame, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probs = logits.softmax(dim=1)
                    
                    # For demonstration, generate placeholder action annotations
                    # In a real implementation, this would map ImageNet classes to actions
                    # actions = ["walking", "running", "sitting", "standing", "talking"]
                    actions = pd.read_csv('/Users/h/Documents/projects_local/life-encoding/scripts/revision/word_corpus/action_words_gerunds_trim.txt', sep='\t', header=None)[0]
                    num_actions = np.random.randint(1, 4)
                    selected_actions = np.random.choice(actions, size=num_actions, replace=False)
                    action_annotations.append(selected_actions.tolist())
            
            else:
                # Default processing for unknown model types
                for _ in range(len(batch_frames)):
                    action_annotations.append(["unknown_action"])
    
    except Exception as e:
        logger.error(f"Error generating action annotations: {str(e)}")
        # Fill in missing annotations
        while len(action_annotations) < len(timestamps):
            action_annotations.append(["error_processing_action"])
    
    # Ensure we have the correct number of annotations
    if len(action_annotations) < len(timestamps):
        logger.warning(f"Action annotation count mismatch: {len(action_annotations)} vs {len(timestamps)}")
        # Fill in missing annotations
        while len(action_annotations) < len(timestamps):
            action_annotations.append(["missing_action"])
    
    return action_annotations[:len(timestamps)]


def generate_agent_annotations(
    frames: List[np.ndarray],
    timestamps: List[float],
    model_dict: Dict,
    batch_size: int = 8,
    device: str = "cpu"
) -> List[List[str]]:
    """
    Generate agent annotations for video frames.
    
    Args:
        frames: List of video frames
        timestamps: List of frame timestamps
        model_dict: Dictionary containing the agent model and processor
        batch_size: Batch size for model inference
        device: Device to use for inference
        
    Returns:
        list: List of agent annotations for each frame
    """
    if not model_dict:
        logger.error("Agent model not provided")
        return [["unknown_agent"] for _ in timestamps]
    
    model = model_dict["model"]
    processor = model_dict["processor"]
    model_type = model_dict["type"]
    
    agent_annotations = []
    
    try:
        # Process frames in batches
        for i in tqdm(range(0, len(frames), batch_size), desc="Generating agent annotations"):
            batch_frames = frames[i:i+batch_size]
            
            if model_type == "clip":
                # CLIP-specific processing for agents
                agent_prompts = pd.read_csv('/Users/h/Documents/projects_local/life-encoding/scripts/revision/word_corpus/agent_words_trim.txt', sep='\t', header=None)[0].tolist()
                # agent_prompts = [
                #     "person", "man", "woman", "child", "dog", "cat", "bird",
                #     "car", "bicycle", "motorcycle", "truck", "boat", "airplane",
                #     "group of people", "crowd", "animal", "robot", "elderly person",
                #     "young adult", "teenager", "baby", "horse", "cow", "sheep"
                # ]
                
                for frame in batch_frames:
                    # Process the frame with CLIP
                    inputs = processor(
                        text=agent_prompts,
                        images=frame,
                        return_tensors="pt",
                        padding=True
                    ).to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        probs = logits_per_image.softmax(dim=1)
                    
                    # Get top 3 agents
                    top_probs, top_indices = probs.topk(3)
                    selected_agents = [agent_prompts[idx] for idx in top_indices[0].tolist()]
                    agent_annotations.append(selected_agents)
            
            elif model_type == "vit":
                # ViT-specific processing for agents
                # This is a simplified implementation
                for frame in batch_frames:
                    # Process the frame with ViT
                    inputs = processor(frame, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probs = logits.softmax(dim=1)
                    
                    # For demonstration, generate placeholder agent annotations
                    # In a real implementation, this would map ImageNet classes to agents
                    # agents = ["person", "dog", "cat", "car", "bicycle"]
                    agents = pd.read_csv('/Users/h/Documents/projects_local/life-encoding/scripts/revision/word_corpus/agent_words_trim.txt', sep='\t', header=None)[0].tolist()
                    num_agents = np.random.randint(1, 4)
                    selected_agents = np.random.choice(agents, size=num_agents, replace=False)
                    agent_annotations.append(selected_agents.tolist())
            
            else:
                # Default processing for unknown model types
                for _ in range(len(batch_frames)):
                    agent_annotations.append(["unknown_agent"])
    
    except Exception as e:
        logger.error(f"Error generating agent annotations: {str(e)}")
        # Fill in missing annotations
        while len(agent_annotations) < len(timestamps):
            agent_annotations.append(["error_processing_agent"])
    
    # Ensure we have the correct number of annotations
    if len(agent_annotations) < len(timestamps):
        logger.warning(f"Agent annotation count mismatch: {len(agent_annotations)} vs {len(timestamps)}")
        # Fill in missing annotations
        while len(agent_annotations) < len(timestamps):
            agent_annotations.append(["missing_agent"])
    
    return agent_annotations[:len(timestamps)]


def generate_scene_annotations(
    frames: List[np.ndarray],
    timestamps: List[float],
    model_dict: Dict,
    batch_size: int = 8,
    device: str = "cpu"
) -> List[List[str]]:
    """
    Generate scene/background annotations for video frames.
    
    Args:
        frames: List of video frames
        timestamps: List of frame timestamps
        model_dict: Dictionary containing the scene model and processor
        batch_size: Batch size for model inference
        device: Device to use for inference
        
    Returns:
        list: List of scene annotations for each frame
    """
    if not model_dict:
        logger.error("Scene model not provided")
        return [["unknown_scene"] for _ in timestamps]
    
    model = model_dict["model"]
    processor = model_dict["processor"]
    model_type = model_dict["type"]
    
    scene_annotations = []
    
    try:
        # Process frames in batches
        for i in tqdm(range(0, len(frames), batch_size), desc="Generating scene annotations"):
            batch_frames = frames[i:i+batch_size]
            
            if model_type == "clip":
                # CLIP-specific processing for scenes
                scene_prompts = pd.read_csv('/Users/h/Documents/projects_local/life-encoding/scripts/revision/word_corpus/scene_words_trim.txt', sep='\t', header=None)[0].tolist()
                #     "indoor", "outdoor", "urban", "rural", "forest", "beach",
                #     "mountain", "desert", "ocean", "lake", "river", "park",
                #     "street", "highway", "building", "house", "apartment",
                #     "office", "restaurant", "store", "school", "hospital",
                #     "airport", "train station", "stadium", "theater", "museum"
                # ]
                
                for frame in batch_frames:
                    # Process the frame with CLIP
                    inputs = processor(
                        text=scene_prompts,
                        images=frame,
                        return_tensors="pt",
                        padding=True
                    ).to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        probs = logits_per_image.softmax(dim=1)
                    
                    # Get top 3 scenes
                    top_probs, top_indices = probs.topk(3)
                    selected_scenes = [scene_prompts[idx] for idx in top_indices[0].tolist()]
                    scene_annotations.append(selected_scenes)
            
            elif model_type == "vit":
                # ViT-specific processing for scenes
                # This is a simplified implementation
                for frame in batch_frames:
                    # Process the frame with ViT
                    inputs = processor(frame, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probs = logits.softmax(dim=1)
                    
                    # For demonstration, generate placeholder scene annotations
                    # In a real implementation, this would map ImageNet classes to scenes
                    # scenes = ["indoor", "outdoor", "urban", "rural", "forest"]
                    scenes = pd.read_csv('/Users/h/Documents/projects_local/life-encoding/scripts/revision/word_corpus/scene_words_trim.txt', sep='\t', header=None)[0].tolist()
                    num_scenes = np.random.randint(1, 4)
                    selected_scenes = np.random.choice(scenes, size=num_scenes, replace=False)
                    scene_annotations.append(selected_scenes.tolist())
            
            else:
                # Default processing for unknown model types
                for _ in range(len(batch_frames)):
                    scene_annotations.append(["unknown_scene"])
    
    except Exception as e:
        logger.error(f"Error generating scene annotations: {str(e)}")
        # Fill in missing annotations
        while len(scene_annotations) < len(timestamps):
            scene_annotations.append(["error_processing_scene"])
    
    # Ensure we have the correct number of annotations
    if len(scene_annotations) < len(timestamps):
        logger.warning(f"Scene annotation count mismatch: {len(scene_annotations)} vs {len(timestamps)}")
        # Fill in missing annotations
        while len(scene_annotations) < len(timestamps):
            scene_annotations.append(["missing_scene"])
    
    return scene_annotations[:len(timestamps)]


def generate_object_annotations(
    frames: List[np.ndarray],
    timestamps: List[float],
    model_dict: Dict,
    batch_size: int = 8,
    device: str = "cpu"
) -> List[List[str]]:
    """
    Generate object annotations for video frames.
    
    Args:
        frames: List of video frames
        timestamps: List of frame timestamps
        model_dict: Dictionary containing the agent model and processor
        batch_size: Batch size for model inference
        device: Device to use for inference
        
    Returns:
        list: List of object annotations for each frame
    """
    if not model_dict:
        logger.error("Object model not provided")
        return [["unknown_object"] for _ in timestamps]
    
    model = model_dict["model"]
    processor = model_dict["processor"]
    model_type = model_dict["type"]
    
    object_annotations = []
    
    try:
        # Process frames in batches
        for i in tqdm(range(0, len(frames), batch_size), desc="Generating object annotations"):
            batch_frames = frames[i:i+batch_size]
            
            if model_type == "clip":
                # CLIP-specific processing for objects
                object_prompts = pd.read_csv('/Users/h/Documents/projects_local/life-encoding/scripts/revision/word_corpus/object_words_trim.txt', sep='\t', header=None)[0].tolist()
                
                for frame in batch_frames:
                    # Process the frame with CLIP
                    inputs = processor(
                        text=object_prompts,
                        images=frame,
                        return_tensors="pt",
                        padding=True
                    ).to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        probs = logits_per_image.softmax(dim=1)
                    
                    # Get top 3 agents
                    top_probs, top_indices = probs.topk(3)
                    selected_objects = [object_prompts[idx] for idx in top_indices[0].tolist()]
                    object_annotations.append(selected_objects)
            
            elif model_type == "vit":
                # ViT-specific processing for agents
                # This is a simplified implementation
                for frame in batch_frames:
                    # Process the frame with ViT
                    inputs = processor(frame, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probs = logits.softmax(dim=1)
                    
                    # For demonstration, generate placeholder agent annotations
                    # In a real implementation, this would map ImageNet classes to agents
                    # agents = ["person", "dog", "cat", "car", "bicycle"]
                    objects = pd.read_csv('/Users/h/Documents/projects_local/life-encoding/scripts/revision/word_corpus/object_words_trim.txt', sep='\t', header=None)[0].tolist()

                    num_objects = np.random.randint(1, 4)
                    selected_objects = np.random.choice(objects, size=num_objects, replace=False)
                    object_annotations.append(selected_objects.tolist())
            
            else:
                # Default processing for unknown model types
                for _ in range(len(batch_frames)):
                    object_annotations.append(["unknown_objects"])
    
    except Exception as e:
        logger.error(f"Error generating object annotations: {str(e)}")
        # Fill in missing annotations
        while len(object_annotations) < len(timestamps):
            object_annotations.append(["error_processing_object"])
    
    # Ensure we have the correct number of annotations
    if len(object_annotations) < len(timestamps):
        logger.warning(f"Object annotation count mismatch: {len(object_annotations)} vs {len(timestamps)}")
        # Fill in missing annotations
        while len(object_annotations) < len(timestamps):
            object_annotations.append(["missing_object"])
    
    return object_annotations[:len(timestamps)]
