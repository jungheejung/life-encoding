#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Loader Module

This module handles loading and initializing the AI models for video annotation:
- CLIP for agent and scene detection
- ViT (Vision Transformer) for general image understanding
- Facebook SlowFast for action recognition
"""

import os
import torch
import logging
from typing import Dict, List, Union, Optional, Any
from transformers import CLIPProcessor, CLIPModel
from transformers import ViTForImageClassification, ViTImageProcessor
import timm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_clip_model(model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu") -> Dict:
    """
    Load a CLIP model for agent and scene detection.
    
    Args:
        model_name: Name of the CLIP model to load
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        dict: Dictionary containing the model and processor
    """
    try:
        logger.info(f"Loading CLIP model: {model_name}")
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        
        device_obj = torch.device(device)
        model = model.to(device_obj)
        model.eval()
        
        return {
            "model": model,
            "processor": processor,
            "name": model_name,
            "type": "clip"
        }
    
    except Exception as e:
        logger.error(f"Error loading CLIP model: {str(e)}")
        return None


def load_vit_model(model_name: str = "google/vit-base-patch16-224", device: str = "cpu") -> Dict:
    """
    Load a Vision Transformer (ViT) model.
    
    Args:
        model_name: Name of the ViT model to load
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        dict: Dictionary containing the model and processor
    """
    try:
        logger.info(f"Loading ViT model: {model_name}")
        model = ViTForImageClassification.from_pretrained(model_name)
        processor = ViTImageProcessor.from_pretrained(model_name)
        
        device_obj = torch.device(device)
        model = model.to(device_obj)
        model.eval()
        
        return {
            "model": model,
            "processor": processor,
            "name": model_name,
            "type": "vit"
        }
    
    except Exception as e:
        logger.error(f"Error loading ViT model: {str(e)}")
        return None


def load_slowfast_model(model_name: str = "facebook/slowfast", device: str = "cpu") -> Dict:
    """
    Load a SlowFast model for action recognition.
    
    Args:
        model_name: Name of the SlowFast model variant
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        dict: Dictionary containing the model and processor
    """
    try:
        logger.info(f"Loading SlowFast model")
        # For SlowFast, we'll use timm or PyTorchVideo
        # This is a simplified implementation - in practice, you'd use PyTorchVideo
        # or a specific SlowFast implementation
        
        # For demonstration, we'll use a placeholder implementation
        model = timm.create_model('resnet50', pretrained=True)
        
        device_obj = torch.device(device)
        model = model.to(device_obj)
        model.eval()
        
        # Define a simple processor function for SlowFast
        def process_frames(frames):
            # This would be replaced with actual SlowFast preprocessing
            return torch.tensor(frames).permute(0, 3, 1, 2).float()
        
        return {
            "model": model,
            "processor": process_frames,
            "name": "facebook/slowfast",
            "type": "slowfast"
        }
    
    except Exception as e:
        logger.error(f"Error loading SlowFast model: {str(e)}")
        return None


def load_models(args) -> Dict:
    """
    Load models based on command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        dict: Dictionary of loaded models
    """
    models = {}
    device = args.device
    
    # If a single model is specified for all categories
    if args.model:
        logger.info(f"Using {args.model} for all annotation categories")
        
        if "clip" in args.model.lower():
            model_dict = load_clip_model(args.model, device)
            if model_dict:
                models["action"] = model_dict
                models["agent"] = model_dict
                models["scene"] = model_dict
        
        elif "vit" in args.model.lower():
            model_dict = load_vit_model(args.model, device)
            if model_dict:
                models["action"] = model_dict
                models["agent"] = model_dict
                models["scene"] = model_dict
        
        elif "slowfast" in args.model.lower():
            model_dict = load_slowfast_model(args.model, device)
            if model_dict:
                models["action"] = model_dict
                models["agent"] = model_dict
                models["scene"] = model_dict
        
        else:
            # Default to CLIP for unknown model types
            logger.warning(f"Unknown model type: {args.model}, defaulting to CLIP")
            model_dict = load_clip_model("openai/clip-vit-base-patch32", device)
            if model_dict:
                models["action"] = model_dict
                models["agent"] = model_dict
                models["scene"] = model_dict
    
    # If category-specific models are specified
    else:
        # Action model
        if args.action_model:
            if "slowfast" in args.action_model.lower():
                models["action"] = load_slowfast_model(args.action_model, device)
            elif "vit" in args.action_model.lower():
                models["action"] = load_vit_model(args.action_model, device)
            elif "clip" in args.action_model.lower():
                models["action"] = load_clip_model(args.action_model, device)
            else:
                models["action"] = load_slowfast_model("facebook/slowfast", device)
        else:
            # Default action model
            models["action"] = load_slowfast_model("facebook/slowfast", device)
        
        # Agent model
        if args.agent_model:
            if "clip" in args.agent_model.lower():
                models["agent"] = load_clip_model(args.agent_model, device)
            elif "vit" in args.agent_model.lower():
                models["agent"] = load_vit_model(args.agent_model, device)
            else:
                models["agent"] = load_clip_model("openai/clip-vit-base-patch32", device)
        else:
            # Default agent model
            models["agent"] = load_clip_model("openai/clip-vit-base-patch32", device)
        
        # Scene model
        if args.scene_model:
            if "clip" in args.scene_model.lower():
                models["scene"] = load_clip_model(args.scene_model, device)
            elif "vit" in args.scene_model.lower():
                models["scene"] = load_vit_model(args.scene_model, device)
            else:
                models["scene"] = load_clip_model("openai/clip-vit-base-patch32", device)
        else:
            # Default scene model
            models["scene"] = load_vit_model("google/vit-base-patch16-224", device)
    
    # Check if all required models are loaded
    if not all(k in models for k in ["action", "agent", "scene"]):
        missing = [k for k in ["action", "agent", "scene"] if k not in models]
        logger.error(f"Missing models for categories: {missing}")
        
        # Load defaults for missing models
        if "action" not in models:
            models["action"] = load_slowfast_model("facebook/slowfast", device)
        if "agent" not in models:
            models["agent"] = load_clip_model("openai/clip-vit-base-patch32", device)
        if "scene" not in models:
            models["scene"] = load_vit_model("google/vit-base-patch16-224", device)
    
    return models


def get_model_info(models: Dict) -> Dict:
    """
    Get information about the loaded models.
    
    Args:
        models: Dictionary of loaded models
        
    Returns:
        dict: Model information
    """
    info = {}
    
    for category, model_dict in models.items():
        if model_dict:
            info[category] = {
                "name": model_dict.get("name", "unknown"),
                "type": model_dict.get("type", "unknown")
            }
    
    return info

