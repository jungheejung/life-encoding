# -*- coding: utf-8 -*-
"""
Video Annotation System for fMRI Research

This script provides a command-line interface for generating comprehensive,
temporally-precise annotations for neuroscience research. It analyzes videos
and produces structured annotations that align with fMRI brain imaging data
at 0.46-second intervals.

Usage:
    python video_annotator.py --input video.mp4 --model microsoft/git-base --sampling-rate 0.46
    python video_annotator.py --input-folder /videos/ --action-model facebook/slowfast --agent-model openai/clip --scene-model google/vit
    python video_annotator.py --input-folder /videos/ --model google/vit-base-patch16-224 --output-dir /annotations/
    python video_annotator.py --input video.mp4 --unified-model clip --output-dir clip_results
    python video_annotator.py --input video.mp4 --generate-transcript --whisper-model openai/whisper-base
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add the parent directory to sys.path to allow imports from the project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_annotation_system.utils.video_utils import validate_input, get_video_metadata
from video_annotation_system.utils.annotation_utils import format_annotations, save_annotations
# from video_annotation_system.utils.transcript_utils import extract_transcript_from_video, align_transcript_with_annotations
from video_annotation_system.models.model_loader import load_models
from video_annotation_system.models.annotation_generator import generate_annotations


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Video Annotation System for fMRI Research"
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=str, help="Path to input video file")
    input_group.add_argument("--input-folder", type=str, help="Path to folder containing videos")
    
    # Model options
    model_group = parser.add_argument_group("Model options")
    model_group.add_argument("--model", type=str, help="HuggingFace model to use (can be combined with --unified-model)")
    model_group.add_argument("--unified-model", type=str, choices=["clip", "vit", "slowfast"],
                            help="Use a single model type for all annotation categories")
    
    category_models = parser.add_argument_group("Category-specific models")
    category_models.add_argument("--action-model", type=str, help="Model for action recognition")
    category_models.add_argument("--agent-model", type=str, help="Model for agent detection")
    category_models.add_argument("--scene-model", type=str, help="Model for scene classification")
    
    # Transcript options
    transcript_group = parser.add_argument_group("Transcript options")
    transcript_group.add_argument("--generate-transcript", action="store_true", 
                                help="Generate transcript from video audio")
    transcript_group.add_argument("--whisper-model", type=str, default="openai/whisper-base",
                                help="Whisper model to use for transcript generation")
    transcript_group.add_argument("--language", type=str, default="en",
                                help="Language code for transcript generation")
    transcript_group.add_argument("--chunk-size", type=int, default=30,
                                help="Size of audio chunks to process (in seconds)")
    transcript_group.add_argument("--overlap", type=int, default=5,
                                help="Overlap between chunks (in seconds)")
    transcript_group.add_argument("--keep-audio", action="store_true",
                                help="Keep extracted audio files")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="./annotations", 
                        help="Directory to save annotation outputs")
    parser.add_argument("--format", type=str, choices=["json", "csv", "pickle"], 
                        default="json", help="Output format")
    
    # Processing options
    parser.add_argument("--sampling-rate", type=float, default=0.46,
                        help="Sampling rate in seconds (default: 0.46)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for model inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference (cuda/cpu)")
    
    return parser.parse_args()


def process_video(video_path, models, args):
    """Process a single video file and generate annotations."""
    print(f"Processing video: {video_path}")
    
    # Validate input video
    if not validate_input(video_path):
        print(f"Error: Could not process {video_path}")
        return None
    
    # Get video metadata
    metadata = get_video_metadata(video_path)
    print(f"Video duration: {metadata['duration']:.2f}s, FPS: {metadata['fps']:.2f}")
    
    # Generate annotations
    start_time = time.time()
    annotations = generate_annotations(
        video_path=video_path,
        models=models,
        sampling_rate=args.sampling_rate,
        batch_size=args.batch_size,
        device=args.device
    )
    
    processing_time = time.time() - start_time
    print(f"Generated {len(annotations)} annotations in {processing_time:.2f}s")
    
    # Generate transcript if requested
    if args.generate_transcript and "whisper" in models:
        print("Transcript generation is currently disabled")
        # print("Generating transcript...")
        # transcript_start_time = time.time()
        
        # transcript_data = extract_transcript_from_video(
        #     video_path=video_path,
        #     model_dict=models["whisper"],
        #     chunk_size=args.chunk_size,
        #     overlap=args.overlap,
        #     language=args.language,
        #     device=args.device,
        #     keep_audio=args.keep_audio
        # )
        
        # transcript_time = time.time() - transcript_start_time
        # print(f"Generated transcript in {transcript_time:.2f}s")
        
        # # Align transcript with annotations
        # annotations = align_transcript_with_annotations(
        #     transcript_data=transcript_data,
        #     annotations=annotations,
        #     sampling_rate=args.sampling_rate
        # )
        
        # print(f"Aligned transcript with {len(annotations)} annotations")
    
    # Format and save annotations
    output_path = save_annotations(
        annotations=annotations,
        video_path=video_path,
        metadata=metadata,
        output_dir=args.output_dir,
        format=args.format
    )
    
    print(f"Saved annotations to {output_path}")
    return output_path


def process_folder(folder_path, models, args):
    """Process all videos in a folder."""
    print(f"Processing videos in folder: {folder_path}")
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    processed_files = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_path = os.path.join(root, file)
                output_path = process_video(video_path, models, args)
                if output_path:
                    processed_files.append(output_path)
    
    print(f"Processed {len(processed_files)} videos")
    return processed_files


def main():
    """Main entry point for the video annotation system."""
    args = parse_arguments()
    
    # Load models
    models = load_models(args)
    
    # Process input (single video or folder)
    if args.input:
        process_video(args.input, models, args)
    elif args.input_folder:
        process_folder(args.input_folder, models, args)
    
    print("Video annotation completed successfully!")


if __name__ == "__main__":
    # Import torch here to avoid importing it when the script is imported as a module
    import torch
    main()

