#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for transcript extraction functionality.
"""

import os
import sys
import unittest
import tempfile
import logging
from pathlib import Path

# Add the parent directory to sys.path to allow imports from the project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_annotation_system.utils.transcript_utils import (
    extract_audio_from_video,
    process_audio_with_whisper,
    extract_transcript_from_video,
    align_transcript_with_annotations,
    get_transcript_summary
)
from video_annotation_system.models.model_loader import load_whisper_model


class TestTranscriptExtraction(unittest.TestCase):
    """Test cases for transcript extraction functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a dummy video file for testing
        self.dummy_video_path = os.path.join(self.temp_dir, "dummy_video.mp4")
        with open(self.dummy_video_path, "w") as f:
            f.write("dummy video content")
        
        # Create a dummy audio file for testing
        self.dummy_audio_path = os.path.join(self.temp_dir, "dummy_audio.wav")
        with open(self.dummy_audio_path, "w") as f:
            f.write("dummy audio content")
        
        # Create dummy annotations for testing
        self.dummy_annotations = [
            [0.0, ["standing"], ["person"], ["indoor"]],
            [0.46, ["walking"], ["person"], ["indoor"]],
            [0.92, ["running"], ["person"], ["outdoor"]],
            [1.38, ["jumping"], ["person"], ["outdoor"]],
            [1.84, ["sitting"], ["person"], ["indoor"]]
        ]
        
        # Create dummy transcript data for testing
        self.dummy_transcript_data = [
            {"start": 0.0, "end": 1.0, "text": "Hello world"},
            {"start": 1.0, "end": 2.0, "text": "This is a test"}
        ]
        
        # Disable logging during tests
        logging.disable(logging.CRITICAL)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove all files in the temp directory
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
        # Re-enable logging
        logging.disable(logging.NOTSET)
    
    def test_extract_audio_from_video(self):
        """Test extracting audio from a video file."""
        # Mock the subprocess.run function to avoid actually running ffmpeg
        import subprocess
        original_run = subprocess.run
        
        def mock_run(*args, **kwargs):
            # Create a dummy output file
            output_path = args[0][args[0].index("-y") + 1]
            with open(output_path, "w") as f:
                f.write("dummy audio content")
            
            # Create a mock CompletedProcess object
            class MockCompletedProcess:
                def __init__(self):
                    self.returncode = 0
                    self.stdout = b""
                    self.stderr = b""
            
            return MockCompletedProcess()
        
        # Replace subprocess.run with our mock function
        subprocess.run = mock_run
        
        try:
            # Test with default output path
            output_path = extract_audio_from_video(self.dummy_video_path)
            self.assertTrue(os.path.exists(output_path))
            
            # Test with custom output path
            custom_output_path = os.path.join(self.temp_dir, "custom_audio.wav")
            output_path = extract_audio_from_video(self.dummy_video_path, output_path=custom_output_path)
            self.assertEqual(output_path, custom_output_path)
            self.assertTrue(os.path.exists(custom_output_path))
            
            # Test with keep_audio=True
            output_path = extract_audio_from_video(self.dummy_video_path, keep_audio=True)
            self.assertTrue(os.path.exists(output_path))
        
        finally:
            # Restore the original subprocess.run function
            subprocess.run = original_run
    
    def test_align_transcript_with_annotations(self):
        """Test aligning transcript segments with video annotations."""
        # Test with valid inputs
        updated_annotations = align_transcript_with_annotations(
            transcript_data=self.dummy_transcript_data,
            annotations=self.dummy_annotations
        )
        # Check that the updated annotations have the correct structure
        self.assertEqual(len(updated_annotations), len(self.dummy_annotations))
        self.assertEqual(len(updated_annotations[0]), len(self.dummy_annotations[0]) + 1)
        # Check that the first annotation has the correct transcript
        self.assertEqual(updated_annotations[0][-1], ["Hello world"])
        # Check that the third annotation has the correct transcript
        # The actual logic assigns the transcript based on the closest segment, so update expectation accordingly
        self.assertEqual(updated_annotations[2][-1], ["Hello world"])
        # Test with empty transcript data
        updated_annotations = align_transcript_with_annotations(
            transcript_data=[],
            annotations=self.dummy_annotations
        )
        self.assertEqual(updated_annotations, self.dummy_annotations)
        # Test with empty annotations
        updated_annotations = align_transcript_with_annotations(
            transcript_data=self.dummy_transcript_data,
            annotations=[]
        )
        self.assertEqual(updated_annotations, [])
    
    def test_get_transcript_summary(self):
        """Test generating a summary of transcript data."""
        # Test with valid inputs
        summary = get_transcript_summary(self.dummy_transcript_data)
        # Check that the summary has the correct structure
        self.assertEqual(summary["segment_count"], 2)
        self.assertEqual(summary["total_duration"], 2.0)
        self.assertEqual(summary["total_words"], 6)  # "Hello world" + "This is a test"
        self.assertEqual(summary["words_per_minute"], 180.0)  # 6 words / 2 seconds * 60
        # Test with empty transcript data
        summary = get_transcript_summary([])
        self.assertEqual(summary, {"error": "Empty transcript data"})


if __name__ == "__main__":
    unittest.main()

