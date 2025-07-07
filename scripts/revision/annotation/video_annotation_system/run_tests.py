#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test runner script for the Video Annotation System.

This script runs all the tests for the Video Annotation System:
- Basic tests (imports, structure)
- Video processing tests
- Model loading tests
- Annotation generation tests
- Annotation formatting tests
- Command-line interface tests
- Performance benchmarks
"""

import os
import sys
import logging
import argparse
import time
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run tests for the Video Annotation System")
    parser.add_argument("--test", type=str, choices=["all", "basic", "video", "model", "annotation", "formatting", "cli", "benchmark"],
                        default="all", help="Which test to run")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to use for model tests")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    return parser.parse_args()


def run_test(test_script, args=None, verbose=False):
    """Run a test script and return the result."""
    command = ["python", test_script]
    if args:
        command.extend(args)
    
    logger.info(f"Running test: {' '.join(command)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            capture_output=not verbose,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"Test passed in {elapsed_time:.2f} seconds")
            return True
        else:
            logger.error(f"Test failed with return code {result.returncode}")
            if not verbose:
                logger.error(f"Stdout: {result.stdout}")
                logger.error(f"Stderr: {result.stderr}")
            return False
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error running test: {str(e)}")
        logger.error(f"Test failed after {elapsed_time:.2f} seconds")
        return False


def run_basic_tests(verbose=False):
    """Run basic tests."""
    logger.info("Running basic tests...")
    return run_test("tests/test_basic.py", verbose=verbose)


def run_video_processing_tests(verbose=False):
    """Run video processing tests."""
    logger.info("Running video processing tests...")
    return run_test("tests/test_video_processing.py", verbose=verbose)


def run_model_loading_tests(device="cpu", verbose=False):
    """Run model loading tests."""
    logger.info(f"Running model loading tests on device: {device}...")
    return run_test("tests/test_model_loading.py", args=["--device", device], verbose=verbose)


def run_annotation_generation_tests(device="cpu", verbose=False):
    """Run annotation generation tests."""
    logger.info(f"Running annotation generation tests on device: {device}...")
    return run_test("tests/test_annotation_generation.py", args=["--device", device], verbose=verbose)


def run_annotation_formatting_tests(verbose=False):
    """Run annotation formatting tests."""
    logger.info("Running annotation formatting tests...")
    return run_test("tests/test_annotation_formatting.py", verbose=verbose)


def run_cli_tests(verbose=False):
    """Run command-line interface tests."""
    logger.info("Running command-line interface tests...")
    return run_test("tests/test_command_line.py", verbose=verbose)


def run_performance_benchmark(device="cpu", verbose=False):
    """Run performance benchmark tests."""
    logger.info(f"Running performance benchmark on device: {device}...")
    
    # Create a test video
    import cv2
    import numpy as np
    
    # Create a test video
    test_video_path = "benchmark_video.mp4"
    duration = 10
    fps = 30
    width = 640
    height = 480
    
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video_path, fourcc, fps, (width, height))
    
    # Generate frames
    total_frames = int(duration * fps)
    
    for i in range(total_frames):
        # Create a frame with a timestamp
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a timestamp
        timestamp = i / fps
        cv2.putText(
            frame, 
            f"Time: {timestamp:.2f}s", 
            (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2
        )
        
        # Add a colored rectangle that changes with time
        color = (
            int(255 * (i / total_frames)),
            int(255 * (1 - i / total_frames)),
            int(127 + 128 * np.sin(i / total_frames * np.pi))
        )
        cv2.rectangle(frame, (width // 4, height // 4), (3 * width // 4, 3 * height // 4), color, -1)
        
        # Write the frame
        out.write(frame)
    
    # Release the VideoWriter
    out.release()
    logger.info(f"Created benchmark video at {test_video_path} ({duration}s, {fps} fps)")
    
    # Run the benchmark
    try:
        # Create output directory
        os.makedirs("benchmark_output", exist_ok=True)
        
        # Run the benchmark with different sampling rates
        sampling_rates = [0.46, 0.23, 0.92]
        batch_sizes = [1, 2, 4]
        
        results = {}
        
        for sampling_rate in sampling_rates:
            for batch_size in batch_sizes:
                key = f"sr{sampling_rate}_bs{batch_size}"
                results[key] = {}
                
                # Run the benchmark
                command = [
                    "python", "video_annotator.py",
                    "--input", test_video_path,
                    "--output-dir", "benchmark_output",
                    "--format", "json",
                    "--sampling-rate", str(sampling_rate),
                    "--batch-size", str(batch_size),
                    "--device", device
                ]
                
                logger.info(f"Running benchmark: {' '.join(command)}")
                
                start_time = time.time()
                
                result = subprocess.run(
                    command,
                    capture_output=not verbose,
                    text=True
                )
                
                elapsed_time = time.time() - start_time
                
                results[key]["elapsed_time"] = elapsed_time
                results[key]["success"] = result.returncode == 0
                
                logger.info(f"Benchmark completed in {elapsed_time:.2f} seconds (sampling_rate={sampling_rate}, batch_size={batch_size})")
        
        # Print benchmark results
        logger.info("\nBenchmark results:")
        logger.info(f"{'Configuration':<20} {'Time (s)':<10} {'Success':<10}")
        logger.info("-" * 40)
        
        for key, data in results.items():
            logger.info(f"{key:<20} {data['elapsed_time']:<10.2f} {data['success']:<10}")
        
        # Clean up
        os.remove(test_video_path)
        
        return all(data["success"] for data in results.values())
    
    except Exception as e:
        logger.error(f"Error in performance benchmark: {str(e)}")
        
        # Clean up
        if os.path.exists(test_video_path):
            os.remove(test_video_path)
        
        return False


def main():
    """Run all tests."""
    args = parse_args()
    
    logger.info("Starting Video Annotation System tests")
    
    test_functions = {
        "basic": lambda: run_basic_tests(args.verbose),
        "video": lambda: run_video_processing_tests(args.verbose),
        "model": lambda: run_model_loading_tests(args.device, args.verbose),
        "annotation": lambda: run_annotation_generation_tests(args.device, args.verbose),
        "formatting": lambda: run_annotation_formatting_tests(args.verbose),
        "cli": lambda: run_cli_tests(args.verbose),
        "benchmark": lambda: run_performance_benchmark(args.device, args.verbose)
    }
    
    if args.test == "all":
        # Run all tests
        all_passed = True
        
        for test_name, test_func in test_functions.items():
            logger.info(f"\n{'='*40}\nRunning {test_name} tests\n{'='*40}")
            result = test_func()
            logger.info(f"{test_name.capitalize()} tests: {'PASSED' if result else 'FAILED'}")
            all_passed = all_passed and result
        
        logger.info(f"\nAll tests: {'PASSED' if all_passed else 'FAILED'}")
    else:
        # Run a specific test
        result = test_functions[args.test]()
        logger.info(f"\n{args.test.capitalize()} tests: {'PASSED' if result else 'FAILED'}")


if __name__ == "__main__":
    main()

