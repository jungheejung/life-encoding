#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for model loading functionality.

This script tests the model loading functions:
- CLIP model loading
- ViT model loading
- SlowFast model loading (or placeholder)
- Model ensemble loading
"""

import os
import sys
import logging
import torch
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test model loading functionality")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to use for model loading")
    return parser.parse_args()


def create_mock_args():
    """Create mock command-line arguments for testing."""
    class MockArgs:
        def __init__(self):
            self.device = "cpu"
            self.model = None
            self.action_model = "facebook/slowfast"
            self.agent_model = "openai/clip-vit-base-patch32"
            self.scene_model = "google/vit-base-patch16-224"
    
    return MockArgs()


def test_clip_model_loading(device="cpu"):
    """Test CLIP model loading."""
    from video_annotation_system.models.model_loader import load_clip_model
    
    logger.info("Testing CLIP model loading...")
    
    # Load a small CLIP model for testing
    model_dict = load_clip_model("openai/clip-vit-base-patch32", device)
    assert model_dict is not None, "Failed to load CLIP model"
    
    # Check model components
    assert "model" in model_dict and "processor" in model_dict, "CLIP model dictionary missing required components"
    
    # Check model type
    assert isinstance(model_dict["model"], torch.nn.Module), "CLIP model is not a torch.nn.Module"
    
    logger.info("CLIP model loaded successfully")
    logger.info(f"Model type: {type(model_dict['model'])}")
    logger.info(f"Processor type: {type(model_dict['processor'])}")


def test_vit_model_loading(device="cpu"):
    """Test ViT model loading."""
    from video_annotation_system.models.model_loader import load_vit_model
    
    logger.info("Testing ViT model loading...")
    
    # Load a small ViT model for testing
    model_dict = load_vit_model("google/vit-base-patch16-224", device)
    assert model_dict is not None, "Failed to load ViT model"
    
    # Check model components
    assert "model" in model_dict and "processor" in model_dict, "ViT model dictionary missing required components"
    
    # Check model type
    assert isinstance(model_dict["model"], torch.nn.Module), "ViT model is not a torch.nn.Module"
    
    logger.info("ViT model loaded successfully")
    logger.info(f"Model type: {type(model_dict['model'])}")
    logger.info(f"Processor type: {type(model_dict['processor'])}")


def test_slowfast_model_loading(device="cpu"):
    """Test SlowFast model loading (or placeholder)."""
    from video_annotation_system.models.model_loader import load_slowfast_model
    
    logger.info("Testing SlowFast model loading...")
    
    # Load a placeholder SlowFast model for testing
    model_dict = load_slowfast_model("facebook/slowfast", device)
    assert model_dict is not None, "Failed to load SlowFast model"
    
    # Check model components
    assert "model" in model_dict and "processor" in model_dict, "SlowFast model dictionary missing required components"
    
    # Check model type
    assert isinstance(model_dict["model"], torch.nn.Module), "SlowFast model is not a torch.nn.Module"
    
    logger.info("SlowFast model (placeholder) loaded successfully")
    logger.info(f"Model type: {type(model_dict['model'])}")
    logger.info(f"Processor type: {type(model_dict['processor'])}")


def test_model_ensemble_loading():
    """Test loading all models as an ensemble."""
    from video_annotation_system.models.model_loader import load_models
    
    logger.info("Testing model ensemble loading...")
    
    # Create mock arguments
    args = create_mock_args()
    
    # Load models
    models = load_models(args)
    assert models, "Failed to load model ensemble"
    
    # Check if all required models are loaded
    required_categories = ["action", "agent", "scene"]
    missing = [category for category in required_categories if category not in models]
    assert not missing, f"Missing models for categories: {missing}"
    
    # Check model info
    model_info = {}
    for category, model_dict in models.items():
        if model_dict:
            model_info[category] = {
                "name": model_dict.get("name", "unknown"),
                "type": model_dict.get("type", "unknown")
            }
    
    logger.info(f"Model ensemble loaded successfully: {model_info}")


def main():
    """Run all tests."""
    args = parse_args()
    device = args.device
    
    logger.info(f"Starting model loading tests on device: {device}")
    
    # Check if CUDA is available when requested
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    tests = [
        ("CLIP model loading", lambda: test_clip_model_loading(device)),
        ("ViT model loading", lambda: test_vit_model_loading(device)),
        ("SlowFast model loading", lambda: test_slowfast_model_loading(device)),
        ("Model ensemble loading", test_model_ensemble_loading)
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


if __name__ == "__main__":
    main()

