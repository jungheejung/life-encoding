#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Loader Module

This module handles loading and initializing the AI models for video annotation:
- CLIP for agent and scene detection
- ViT (Vision Transformer) for general image understanding
- Facebook SlowFast for action recognition
- Whisper for speech recognition and transcript generation
"""

import os
import torch
import logging
import numpy as np
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


def load_slowfast_model(model_name: str = "slowfast_r50", device: str = "cpu") -> Dict:
    """
    Load a SlowFast model for action recognition.
    
    Args:
        model_name: Name of the SlowFast model variant (e.g., 'slowfast_r50', 'slowfast_r101')
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        dict: Dictionary containing the model and processor
    """
    try:
        logger.info(f"Loading SlowFast model: {model_name}")
        
        # Import PyTorchVideo modules
        try:
            import pytorchvideo
            from pytorchvideo.models.hub import slowfast_r50, slowfast_r101
            from pytorchvideo.transforms import (
                ApplyTransformToKey,
                ShortSideScale,
                UniformTemporalSubsample,
                Normalize,
            )
            from torchvision.transforms import Compose, Lambda
            import torch.nn.functional as F
        except ImportError:
            logger.error("PyTorchVideo not installed. Please install with: pip install pytorchvideo")
            raise ImportError("PyTorchVideo not installed")
        
        # Load the appropriate SlowFast model
        if model_name == "slowfast_r50" or model_name == "facebook/slowfast":
            model = slowfast_r50(pretrained=True)
        elif model_name == "slowfast_r101":
            model = slowfast_r101(pretrained=True)
        else:
            logger.warning(f"Unknown SlowFast model: {model_name}, defaulting to slowfast_r50")
            model = slowfast_r50(pretrained=True)
        
        device_obj = torch.device(device)
        model = model.to(device_obj)
        model.eval()
        
        # Load Kinetics 400 class names
        kinetics_classnames = [
            "abseiling", "air drumming", "answering questions", "applauding", "applying cream", "archery",
            "arm wrestling", "arranging flowers", "assembling computer", "auctioning", "baby waking up", "baking cookies",
            "balloon blowing", "bandaging", "barbequing", "bartending", "beatboxing", "bee keeping", "belly dancing",
            "bench pressing", "bending back", "bending metal", "biking through snow", "blasting sand", "blowing glass",
            "blowing nose", "blowing out candles", "bobsledding", "bookbinding", "bouncing on trampoline", "bowling",
            "braiding hair", "breading or breadcrumbing", "breakdancing", "brush painting", "brushing hair",
            "brushing teeth", "building cabinet", "building shed", "bungee jumping", "busking", "canoeing or kayaking",
            "capoeira", "carrying baby", "cartwheeling", "carving pumpkin", "catching fish", "catching or throwing baseball",
            "catching or throwing frisbee", "catching or throwing softball", "celebrating", "changing oil", "changing wheel",
            "checking tires", "cheerleading", "chopping wood", "clapping", "clay pottery making", "clean and jerk",
            "cleaning floor", "cleaning gutters", "cleaning pool", "cleaning shoes", "cleaning toilet", "cleaning windows",
            "climbing a rope", "climbing ladder", "climbing tree", "contact juggling", "cooking chicken", "cooking egg",
            "cooking on campfire", "cooking sausages", "counting money", "country line dancing", "cracking neck", "crawling baby",
            "crossing river", "crying", "curling hair", "cutting nails", "cutting pineapple", "cutting watermelon", "dancing ballet",
            "dancing charleston", "dancing gangnam style", "dancing macarena", "deadlifting", "decorating the christmas tree",
            "digging", "dining", "disc golfing", "diving cliff", "dodgeball", "doing aerobics", "doing laundry", "doing nails",
            "drawing", "dribbling basketball", "drinking", "drinking beer", "drinking shots", "driving car", "driving tractor",
            "drop kicking", "drumming fingers", "dunking basketball", "dying hair", "eating burger", "eating cake",
            "eating carrots", "eating chips", "eating doughnuts", "eating hotdog", "eating ice cream", "eating spaghetti",
            "eating watermelon", "egg hunting", "exercising arm", "exercising with an exercise ball", "extinguishing fire",
            "faceplanting", "feeding birds", "feeding fish", "feeding goats", "filling eyebrows", "finger snapping",
            "fixing hair", "flipping pancake", "flying kite", "folding clothes", "folding napkins", "folding paper",
            "front raises", "frying vegetables", "garbage collecting", "gargling", "getting a haircut", "getting a tattoo",
            "giving or receiving award", "golf chipping", "golf driving", "golf putting", "grinding meat", "grooming dog",
            "grooming horse", "gymnastics tumbling", "hammer throw", "headbanging", "headbutting", "high jump", "high kick",
            "hitting baseball", "hockey stop", "holding snake", "hopscotch", "hoverboarding", "hugging", "hula hooping",
            "hurdling", "hurling (sport)", "ice climbing", "ice fishing", "ice skating", "ironing", "javelin throw",
            "jetskiing", "jogging", "juggling balls", "juggling fire", "juggling soccer ball", "jumping into pool",
            "jumpstyle dancing", "kicking field goal", "kicking soccer ball", "kissing", "kitesurfing", "knitting",
            "krumping", "laughing", "laying bricks", "long jump", "lunge", "making a cake", "making a sandwich", "making bed",
            "making jewelry", "making pizza", "making snowman", "making sushi", "making tea", "marching", "massaging back",
            "massaging feet", "massaging legs", "massaging person's head", "milking cow", "mopping floor", "motorcycling",
            "moving furniture", "mowing lawn", "news anchoring", "opening bottle", "opening present", "paragliding",
            "parasailing", "parkour", "passing American football (in game)", "passing American football (not in game)",
            "peeling apples", "peeling potatoes", "petting animal (not cat)", "petting cat", "picking fruit", "planting trees",
            "plastering", "playing accordion", "playing badminton", "playing bagpipes", "playing basketball", "playing bass guitar",
            "playing cards", "playing cello", "playing chess", "playing clarinet", "playing controller", "playing cricket",
            "playing cymbals", "playing didgeridoo", "playing drums", "playing flute", "playing guitar", "playing harmonica",
            "playing harp", "playing ice hockey", "playing keyboard", "playing kickball", "playing monopoly", "playing organ",
            "playing paintball", "playing piano", "playing poker", "playing recorder", "playing saxophone", "playing squash or racquetball",
            "playing tennis", "playing trombone", "playing trumpet", "playing ukulele", "playing violin", "playing volleyball",
            "playing xylophone", "pole vault", "presenting weather forecast", "pull ups", "pumping fist", "pumping gas",
            "punching bag", "punching person (boxing)", "push up", "pushing car", "pushing cart", "pushing wheelchair",
            "reading book", "reading newspaper", "recording music", "riding a bike", "riding camel", "riding elephant",
            "riding mechanical bull", "riding mountain bike", "riding mule", "riding or walking with horse", "riding scooter",
            "riding unicycle", "ripping paper", "robot dancing", "rock climbing", "rock scissors paper", "roller skating",
            "running on treadmill", "sailing", "salsa dancing", "sanding floor", "scrambling eggs", "scuba diving",
            "setting table", "shaking hands", "shaking head", "sharpening knives", "sharpening pencil", "shaving head",
            "shaving legs", "shearing sheep", "shining shoes", "shooting basketball", "shooting goal (soccer)",
            "shot put", "shoveling snow", "shredding paper", "shuffling cards", "side kick", "sign language interpreting",
            "singing", "situp", "skateboarding", "ski jumping", "skiing (not slalom or crosscountry)", "skiing crosscountry",
            "skiing slalom", "skipping rope", "skydiving", "slacklining", "slapping", "sled dog racing", "smoking",
            "smoking hookah", "snatch weight lifting", "sneezing", "sniffing", "snorkeling", "snowboarding", "snowkiting",
            "snowmobiling", "somersaulting", "spinning poi", "spray painting", "spraying", "springboard diving", "squat",
            "sticking tongue out", "stomping grapes", "stretching arm", "stretching leg", "strumming guitar", "surfing crowd",
            "surfing water", "sweeping floor", "swimming backstroke", "swimming breast stroke", "swimming butterfly stroke",
            "swing dancing", "swinging legs", "swinging on something", "sword fighting", "tai chi", "taking a shower",
            "tango dancing", "tap dancing", "tapping guitar", "tapping pen", "tasting beer", "tasting food", "testifying",
            "texting", "throwing axe", "throwing ball", "throwing discus", "tickling", "tobogganing", "tossing coin",
            "tossing salad", "training dog", "trapezing", "trimming or shaving beard", "trimming trees", "triple jump",
            "tying bow tie", "tying knot (not on a tie)", "tying tie", "unboxing", "unloading truck", "using computer",
            "using remote controller (not gaming)", "using segway", "vault", "waiting in line", "walking the dog",
            "washing dishes", "washing feet", "washing hair", "washing hands", "water skiing", "water sliding", "watering plants",
            "waxing back", "waxing chest", "waxing eyebrows", "waxing legs", "weaving basket", "welding", "whistling",
            "windsurfing", "wrapping present", "wrestling", "writing", "yawning", "yoga", "zumba"
        ]
        
        # Define a processor function for SlowFast
        # SlowFast requires specific preprocessing for the slow and fast pathways
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 32  # SlowFast typically uses 32 frames
        sampling_rate = 2
        frames_per_second = 30
        alpha = 4  # For SlowFast model
        
        class PackPathway(torch.nn.Module):
            """
            Transform for converting video frames into SlowFast pathway inputs.
            """
            def __init__(self, alpha=4):
                super().__init__()
                self.alpha = alpha
            
            def forward(self, frames):
                fast_pathway = frames
                # Perform temporal sampling from the fast pathway to generate the slow pathway
                slow_pathway = torch.index_select(
                    frames,
                    1,
                    torch.linspace(
                        0, frames.shape[1] - 1, frames.shape[1] // self.alpha
                    ).long(),
                )
                frame_list = [slow_pathway, fast_pathway]
                return frame_list
        
        transform = Compose([
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    ShortSideScale(size=side_size),
                    PackPathway(alpha=alpha)
                ]),
            )
        ])
        
        def process_frames(frames):
            """
            Process frames for SlowFast model input.
            
            Args:
                frames: List of numpy arrays with shape (H, W, C)
                
            Returns:
                Processed frames in the format expected by SlowFast
            """
            # Convert frames to tensor
            if len(frames) < num_frames:
                # If we have fewer frames than needed, duplicate the last frame
                frames = frames + [frames[-1]] * (num_frames - len(frames))
            elif len(frames) > num_frames:
                # If we have more frames than needed, sample uniformly
                indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
                frames = [frames[i] for i in indices]
            
            # Convert to tensor with shape (T, H, W, C)
            frames_tensor = torch.tensor(np.array(frames))
            # Permute to (C, T, H, W) as expected by PyTorchVideo
            frames_tensor = frames_tensor.permute(3, 0, 1, 2)
            
            # Apply transform
            input_dict = {"video": frames_tensor}
            transformed_video = transform(input_dict)["video"]
            
            # Move to device
            transformed_video = [pathway.to(device) for pathway in transformed_video]
            
            return transformed_video
        
        # Create a function to map model outputs to action labels
        def map_outputs_to_labels(outputs, top_k=5):
            """
            Map model outputs to human-readable action labels.
            
            Args:
                outputs: Model output tensor
                top_k: Number of top predictions to return
                
            Returns:
                List of top-k predicted action labels
            """
            # Get probabilities
            probs = F.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = probs.topk(top_k)
            
            # Map indices to class names
            predictions = []
            for i in range(top_k):
                idx = top_indices[0, i].item()
                if idx < len(kinetics_classnames):
                    predictions.append(kinetics_classnames[idx])
                else:
                    predictions.append(f"unknown_action_{idx}")
            
            return predictions
        
        return {
            "model": model,
            "processor": process_frames,
            "output_mapper": map_outputs_to_labels,
            "name": model_name,
            "type": "slowfast"
        }
    
    except Exception as e:
        logger.error(f"Error loading SlowFast model: {str(e)}")
        return None


def load_whisper_model(model_name: str = "openai/whisper-base", device: str = "cpu") -> Dict:
    """
    Load a Whisper model for speech recognition and transcript generation.
    
    Args:
        model_name: Name of the Whisper model to load
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        dict: Dictionary containing the model and processor
    """
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        logger.info(f"Loading Whisper model: {model_name}")
        
        # Load the model and processor
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        # Move model to the specified device
        device_obj = torch.device(device)
        model = model.to(device_obj)
        model.eval()
        
        return {
            "model": model,
            "processor": processor,
            "name": model_name,
            "type": "whisper"
        }
    
    except Exception as e:
        logger.error(f"Error loading Whisper model: {str(e)}")
        return None


def load_unified_model(model_type: str, model_name: str = None, device: str = "cpu") -> Dict:
    """
    Load a single model to be used for all annotation categories.
    
    Args:
        model_type: Type of model to load ('clip', 'vit', or 'slowfast')
        model_name: Name of the model to load (if None, uses default for the model type)
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        dict: Dictionary containing the model and processor
    """
    logger.info(f"Loading unified model of type {model_type}")
    
    if model_type.lower() == "clip":
        model_name = model_name or "openai/clip-vit-base-patch32"
        model_dict = load_clip_model(model_name, device)
    elif model_type.lower() == "vit":
        model_name = model_name or "google/vit-base-patch16-224"
        model_dict = load_vit_model(model_name, device)
    elif model_type.lower() == "slowfast":
        model_name = model_name or "facebook/slowfast"
        model_dict = load_slowfast_model(model_name, device)
    else:
        logger.warning(f"Unknown model type: {model_type}, defaulting to CLIP")
        model_name = model_name or "openai/clip-vit-base-patch32"
        model_dict = load_clip_model(model_name, device)
    
    if model_dict:
        model_dict["unified"] = True
    
    return model_dict


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
    
    # If unified model is specified
    if hasattr(args, 'unified_model') and args.unified_model:
        logger.info(f"Using unified model: {args.unified_model}")
        model_dict = load_unified_model(args.unified_model, args.model, device)
        if model_dict:
            models["unified"] = model_dict
            # Also set individual category models to the same model for compatibility
            models["action"] = model_dict
            models["agent"] = model_dict
            models["scene"] = model_dict
    
    # If a single model is specified for all categories
    elif args.model:
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
    
    # Load Whisper model for transcript generation if requested
    if hasattr(args, 'generate_transcript') and args.generate_transcript:
        logger.info("Loading Whisper model for transcript generation")
        whisper_model = args.whisper_model if hasattr(args, 'whisper_model') and args.whisper_model else "openai/whisper-base"
        models["whisper"] = load_whisper_model(whisper_model, device)
    
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

