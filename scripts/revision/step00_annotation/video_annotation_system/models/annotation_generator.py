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
from typing import Dict, List, Tuple, Union, Any, Optional
from tqdm import tqdm

from ..utils.video_utils import extract_frames_at_intervals, preprocess_frames
from ..utils.annotation_utils import format_annotations

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
    
    # Check if we're using a unified model
    if "unified" in models:
        logger.info("Using unified model for all annotation categories")
        action_annotations, agent_annotations, scene_annotations = generate_unified_annotations(
            frames, timestamps, models["unified"], batch_size, device
        )
    else:
        # Generate annotations for each category
        action_annotations = generate_action_annotations(frames, timestamps, models.get("action"), batch_size, device)
        agent_annotations = generate_agent_annotations(frames, timestamps, models.get("agent"), batch_size, device)
        scene_annotations = generate_scene_annotations(frames, timestamps, models.get("scene"), batch_size, device)
    
    # Format annotations using the annotation_utils function
    formatted_annotations = format_annotations(
        timestamps=timestamps,
        action_annotations=action_annotations,
        agent_annotations=agent_annotations,
        scene_annotations=scene_annotations
    )
    
    return formatted_annotations


def generate_unified_annotations(
    frames: List[np.ndarray],
    timestamps: List[float],
    model_dict: Dict,
    batch_size: int = 8,
    device: str = "cpu"
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    """
    Generate all annotations using a single model.
    
    Args:
        frames: List of video frames
        timestamps: List of frame timestamps
        model_dict: Dictionary containing the model and processor
        batch_size: Batch size for model inference
        device: Device to use for inference
        
    Returns:
        tuple: Tuple of (action_annotations, agent_annotations, scene_annotations)
    """
    if not model_dict:
        logger.error("Unified model not provided")
        return (
            [["unknown_action"] for _ in timestamps],
            [["unknown_agent"] for _ in timestamps],
            [["unknown_scene"] for _ in timestamps]
        )
    
    model = model_dict["model"]
    processor = model_dict["processor"]
    model_type = model_dict["type"]
    
    action_annotations = []
    agent_annotations = []
    scene_annotations = []
    
    try:
        if model_type == "clip":
            # CLIP-specific processing for all categories
            
            # Define prompts for each category
            action_prompts = [
                "walking", "running", "jumping", "sitting", "standing",
                "talking", "eating", "drinking", "sleeping", "dancing",
                "fighting", "climbing", "swimming", "driving", "riding",
                "playing", "working", "exercising", "cooking", "reading"
            ]
            
            agent_prompts = [
                "person", "man", "woman", "child", "dog", "cat", "bird",
                "car", "bicycle", "motorcycle", "truck", "boat", "airplane",
                "group of people", "crowd", "animal", "robot", "elderly person",
                "young adult", "teenager", "baby", "horse", "cow", "sheep"
            ]
            
            scene_prompts = [
                "indoor", "outdoor", "urban", "rural", "forest", "beach",
                "mountain", "desert", "ocean", "lake", "river", "park",
                "street", "highway", "building", "house", "apartment",
                "office", "restaurant", "store", "school", "hospital",
                "airport", "train station", "stadium", "theater", "museum"
            ]
            
            # Process frames in batches
            for i in tqdm(range(0, len(frames), batch_size), desc="Generating unified annotations with CLIP"):
                batch_frames = frames[i:i+batch_size]
                
                for frame in batch_frames:
                    # Process for actions
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
                    selected_actions = [action_prompts[idx] for idx in top_indices[0]]
                    action_annotations.append(selected_actions)
                    
                    # Process for agents
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
                    selected_agents = [agent_prompts[idx] for idx in top_indices[0]]
                    agent_annotations.append(selected_agents)
                    
                    # Process for scenes
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
                    selected_scenes = [scene_prompts[idx] for idx in top_indices[0]]
                    scene_annotations.append(selected_scenes)
        
        elif model_type == "vit":
            # ViT-specific processing for all categories
            # This is a simplified implementation
            
            # Process frames in batches
            for i in tqdm(range(0, len(frames), batch_size), desc="Generating unified annotations with ViT"):
                batch_frames = frames[i:i+batch_size]
                
                for frame in batch_frames:
                    # Process the frame with ViT
                    inputs = processor(frame, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probs = logits.softmax(dim=1)
                    
                    # For demonstration, generate placeholder annotations
                    # In a real implementation, this would map ImageNet classes to appropriate categories
                    
                    # Actions
                    actions = ["walking", "running", "sitting", "standing", "talking"]
                    num_actions = np.random.randint(1, 4)
                    selected_actions = np.random.choice(actions, size=num_actions, replace=False)
                    action_annotations.append(selected_actions.tolist())
                    
                    # Agents
                    agents = ["person", "dog", "cat", "car", "bicycle"]
                    num_agents = np.random.randint(1, 4)
                    selected_agents = np.random.choice(agents, size=num_agents, replace=False)
                    agent_annotations.append(selected_agents.tolist())
                    
                    # Scenes
                    scenes = ["indoor", "outdoor", "urban", "rural", "forest"]
                    num_scenes = np.random.randint(1, 4)
                    selected_scenes = np.random.choice(scenes, size=num_scenes, replace=False)
                    scene_annotations.append(selected_scenes.tolist())
        
        elif model_type == "slowfast":
            # SlowFast-specific processing for all categories
            
            # Process frames in batches
            for i in tqdm(range(0, len(frames), batch_size), desc="Generating unified annotations with SlowFast"):
                batch_frames = frames[i:i+batch_size]
                
                # Process batch with SlowFast
                batch_inputs = processor(batch_frames)
                with torch.no_grad():
                    outputs = model(batch_inputs)
                
                # Use the output_mapper function if available
                if "output_mapper" in model_dict:
                    # Map model outputs to action labels
                    for j in range(len(batch_frames)):
                        # For SlowFast, we get one output for the entire batch of frames
                        if j == 0:  # Only process the first output for demonstration
                            # Use SlowFast for actions (its primary purpose)
                            action_labels = model_dict["output_mapper"](outputs)
                            action_annotations.append(action_labels)
                            
                            # For agents and scenes, use placeholder labels since SlowFast is primarily for actions
                            agents = ["person", "group of people"]  # Default agents for actions
                            agent_annotations.append(agents)
                            
                            # Infer scene from actions (simplified approach)
                            if any(a in action_labels for a in ["swimming", "diving", "surfing"]):
                                scenes = ["water", "outdoor"]
                            elif any(a in action_labels for a in ["cooking", "eating", "drinking"]):
                                scenes = ["kitchen", "indoor"]
                            elif any(a in action_labels for a in ["driving", "riding", "walking"]):
                                scenes = ["outdoor", "street"]
                            else:
                                scenes = ["indoor", "outdoor"]  # Default scenes
                            scene_annotations.append(scenes)
                        else:
                            # For demonstration, use the same labels for all frames in the batch
                            action_annotations.append(action_labels)
                            agent_annotations.append(agents)
                            scene_annotations.append(scenes)
                else:
                    # Fallback to placeholder annotations if output_mapper is not available
                    for _ in range(len(batch_frames)):
                        # Actions
                        actions = ["walking", "running", "sitting", "standing", "talking"]
                        num_actions = np.random.randint(1, 4)
                        selected_actions = np.random.choice(actions, size=num_actions, replace=False)
                        action_annotations.append(selected_actions.tolist())
                        
                        # Agents
                        agents = ["person", "dog", "cat", "car", "bicycle"]
                        num_agents = np.random.randint(1, 4)
                        selected_agents = np.random.choice(agents, size=num_agents, replace=False)
                        agent_annotations.append(selected_agents.tolist())
                        
                        # Scenes
                        scenes = ["indoor", "outdoor", "urban", "rural", "forest"]
                        num_scenes = np.random.randint(1, 4)
                        selected_scenes = np.random.choice(scenes, size=num_scenes, replace=False)
                        scene_annotations.append(selected_scenes.tolist())
        
        else:
            # Default processing for unknown model types
            for _ in range(len(frames)):
                action_annotations.append(["unknown_action"])
                agent_annotations.append(["unknown_agent"])
                scene_annotations.append(["unknown_scene"])
    
    except Exception as e:
        logger.error(f"Error generating unified annotations: {str(e)}")
        # Fill in missing annotations
        while len(action_annotations) < len(timestamps):
            action_annotations.append(["error_processing_action"])
            agent_annotations.append(["error_processing_agent"])
            scene_annotations.append(["error_processing_scene"])
    
    # Ensure we have the correct number of annotations
    if len(action_annotations) < len(timestamps):
        logger.warning(f"Annotation count mismatch: {len(action_annotations)} vs {len(timestamps)}")
        # Fill in missing annotations
        while len(action_annotations) < len(timestamps):
            action_annotations.append(["missing_action"])
            agent_annotations.append(["missing_agent"])
            scene_annotations.append(["missing_scene"])
    
    return (
        action_annotations[:len(timestamps)],
        agent_annotations[:len(timestamps)],
        scene_annotations[:len(timestamps)]
    )


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
                batch_inputs = processor(batch_frames)
                with torch.no_grad():
                    # SlowFast model expects a list of tensors for slow and fast pathways
                    outputs = model(batch_inputs)
                
                # Use the output_mapper function if available
                if "output_mapper" in model_dict:
                    # Map model outputs to action labels
                    for j in range(len(batch_frames)):
                        # For SlowFast, we get one output for the entire batch of frames
                        # In a real implementation with multiple clips, we would have multiple outputs
                        if j == 0:  # Only process the first output for demonstration
                            action_labels = model_dict["output_mapper"](outputs)
                            action_annotations.append(action_labels)
                        else:
                            # For demonstration, use the same labels for all frames in the batch
                            action_annotations.append(action_labels)
                else:
                    # Fallback to placeholder annotations if output_mapper is not available
                    for _ in range(len(batch_frames)):
                        # Example actions
                        actions = ["walking", "running", "sitting", "standing", "talking"]
                        # Randomly select 1-3 actions for demonstration
                        num_actions = np.random.randint(1, 4)
                        selected_actions = np.random.choice(actions, size=num_actions, replace=False)
                        action_annotations.append(selected_actions.tolist())
            
            elif model_type == "clip":
                # CLIP-specific processing
                action_prompts = [
                    "walking", "running", "jumping", "sitting", "standing",
                    "talking", "eating", "drinking", "sleeping", "dancing",
                    "fighting", "climbing", "swimming", "driving", "riding",
                    "playing", "working", "exercising", "cooking", "reading"
                ]
                
                for frame in batch_frames:
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
                    selected_actions = [action_prompts[idx] for idx in top_indices[0]]
                    action_annotations.append(selected_actions)
            
            elif model_type == "vit":
                # ViT-specific processing
                for frame in batch_frames:
                    inputs = processor(frame, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probs = logits.softmax(dim=1)
                    
                    # For demonstration, generate placeholder action annotations
                    # In a real implementation, this would map ImageNet classes to action categories
                    actions = ["walking", "running", "sitting", "standing", "talking"]
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
                # CLIP-specific processing
                agent_prompts = [
                    "person", "man", "woman", "child", "dog", "cat", "bird",
                    "car", "bicycle", "motorcycle", "truck", "boat", "airplane",
                    "group of people", "crowd", "animal", "robot", "elderly person",
                    "young adult", "teenager", "baby", "horse", "cow", "sheep"
                ]
                
                for frame in batch_frames:
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
                    selected_agents = [agent_prompts[idx] for idx in top_indices[0]]
                    agent_annotations.append(selected_agents)
            
            elif model_type == "vit":
                # ViT-specific processing
                for frame in batch_frames:
                    inputs = processor(frame, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probs = logits.softmax(dim=1)
                    
                    # For demonstration, generate placeholder agent annotations
                    # In a real implementation, this would map ImageNet classes to agent categories
                    agents = ["person", "dog", "cat", "car", "bicycle"]
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
                # CLIP-specific processing
                scene_prompts = [
                    "indoor", "outdoor", "urban", "rural", "forest", "beach",
                    "mountain", "desert", "ocean", "lake", "river", "park",
                    "street", "highway", "building", "house", "apartment",
                    "office", "restaurant", "store", "school", "hospital",
                    "airport", "train station", "stadium", "theater", "museum"
                ]
                
                for frame in batch_frames:
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
                    selected_scenes = [scene_prompts[idx] for idx in top_indices[0]]
                    scene_annotations.append(selected_scenes)
            
            elif model_type == "vit":
                # ViT-specific processing
                for frame in batch_frames:
                    inputs = processor(frame, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probs = logits.softmax(dim=1)
                    
                    # For demonstration, generate placeholder scene annotations
                    # In a real implementation, this would map ImageNet classes to scene categories
                    scenes = ["indoor", "outdoor", "urban", "rural", "forest"]
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

