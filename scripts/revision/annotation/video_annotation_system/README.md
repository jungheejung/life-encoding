# Multi-Category Video Annotation System for fMRI Research

A Python-based system for automated video annotation that produces structured annotations at precise 0.46-second intervals using state-of-the-art AI models (CLIP, ViT, Facebook SlowFast, and Whisper).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Advanced Options](#advanced-options)
  - [Transcript Generation](#transcript-generation)
  - [Batch Processing](#batch-processing)
- [Output Formats](#output-formats)
- [Models](#models)
- [Validation Metrics](#validation-metrics)
- [Performance Benchmarks](#performance-benchmarks)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Multi-Category Video Annotation System is designed specifically for fMRI research, providing automated annotations of video content at precise 0.46-second intervals. The system analyzes videos using an ensemble of state-of-the-art AI models to extract annotations in three categories:

1. **Actions**: What activities are occurring in the video (e.g., walking, running, talking)
2. **Agents**: Who or what is performing the actions (e.g., person, dog, car)
3. **Backgrounds**: Where the action is taking place (e.g., indoor, outdoor, kitchen)
4. **Transcripts**: What is being said in the video (speech-to-text)

These annotations can be used to correlate with fMRI data, enabling researchers to study brain responses to different visual and auditory stimuli with high temporal precision.

## Features

- **Precise Temporal Sampling**: Extracts frames and generates annotations at exact 0.46-second intervals
- **Multi-Category Annotation**: Provides annotations for actions, agents, backgrounds, and transcripts
- **State-of-the-Art Models**: Uses CLIP, ViT, Facebook SlowFast, and Whisper models for high-quality annotations
- **Speech Recognition**: Extracts and transcribes speech from videos using OpenAI's Whisper model
- **Multiple Output Formats**: Supports JSON, CSV, and pickle output formats
- **Batch Processing**: Process multiple videos in a single command
- **Validation Metrics**: Calculate metrics to evaluate annotation quality
- **Performance Benchmarking**: Measure system performance with different configurations
- **Comprehensive Testing**: Includes unit tests, integration tests, and a test runner

## System Architecture

The system follows a modular architecture with the following components:

1. **Video Processing**: Handles video loading, validation, and frame extraction
2. **Audio Processing**: Extracts audio from videos for transcript generation
3. **Model Integration**: Loads and manages AI models for annotation generation
4. **Annotation Generation**: Generates annotations from video frames using AI models
5. **Transcript Generation**: Generates transcripts from audio using Whisper model
6. **Output Formatting**: Formats and saves annotations in various formats
7. **Command-Line Interface**: Provides a user-friendly interface for system operation
8. **Testing Framework**: Ensures system reliability and performance
9. **Validation Metrics**: Evaluates annotation quality and consistency

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Video Input    │────▶│  Frame          │────▶│  Model          │
│  - Single file  │     │  Extraction     │     │  Integration    │
│  - Folder       │     │  (0.46s)        │     │  (CLIP, ViT,    │
└─────────┬───────┘     └─────────────────┘     │   SlowFast)     │
          │                                     └────────┬────────┘
          │                                              │
          │             ┌─────────────────┐     ┌───────▼────────┐
          └────────────▶│  Audio          │────▶│  Whisper       │
                        │  Extraction     │     │  Model         │
                        └─────────────────┘     └───────┬────────┘
                                                        │
┌─────────────────┐     ┌─────────────────┐     ┌──────▼─────────┐
│  Output         │◀────│  Annotation     │◀────│  Annotation    │
│  - JSON         │     │  Formatting     │     │  Generation    │
│  - CSV          │     │  - Actions      │     │  - Actions     │
│  - Pickle       │     │  - Agents       │     │  - Agents      │
└─────────────────┘     │  - Backgrounds  │     │  - Backgrounds │
                        │  - Transcripts  │     │  - Transcripts │
                        └─────────────────┘     └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/video-annotation-system.git
   cd video-annotation-system
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies using uv (recommended) or pip:
   ```bash
   # Using uv (faster)
   uv venv
   uv pip install -r requirements.txt
   
   # Using pip
   pip install -r requirements.txt
   ```

4. Install the package in development mode:
   ```bash
   uv pip install -e .  # or: pip install -e .
   ```

## Usage

### Basic Usage

To annotate a single video:

```bash
python video_annotator.py --input path/to/video.mp4 --output-dir path/to/output
```

### Advanced Options

You can specify different models for each annotation category:

```bash
python video_annotator.py --input path/to/video.mp4 \
                         --action-model facebook/slowfast \
                         --agent-model openai/clip-vit-base-patch32 \
                         --scene-model google/vit-base-patch16-224 \
                         --output-dir path/to/output
```

Or use a single model for all categories:

```bash
python video_annotator.py --input path/to/video.mp4 \
                         --unified-model clip \
                         --output-dir path/to/output
```

### Transcript Generation

To generate transcripts from video audio:

```bash
python video_annotator.py --input path/to/video.mp4 \
                         --generate-transcript \
                         --whisper-model openai/whisper-base \
                         --output-dir path/to/output
```

Additional transcript options:

```bash
python video_annotator.py --input path/to/video.mp4 \
                         --generate-transcript \
                         --whisper-model openai/whisper-small \
                         --language en \
                         --chunk-size 30 \
                         --overlap 5 \
                         --keep-audio \
                         --output-dir path/to/output
```

### Batch Processing

To process all videos in a folder:

```bash
python video_annotator.py --input-folder path/to/videos/ \
                         --generate-transcript \
                         --output-dir path/to/output
```

## Output Formats

The system supports three output formats:

1. **JSON** (default): Structured format with metadata and annotations
2. **CSV**: Simplified format for easy import into spreadsheets
3. **Pickle**: Binary format for Python applications

Example JSON output:

```json
{
  "metadata": {
    "video_filename": "example.mp4",
    "duration": 120.5,
    "fps": 30.0,
    "frame_count": 3615,
    "width": 1920,
    "height": 1080,
    "sampling_rate": 0.46,
    "annotation_count": 262,
    "has_transcript": true
  },
  "annotations": [
    [
      0.0,
      ["standing", "talking"],
      ["person", "woman"],
      ["indoor", "office"],
      ["Hello, welcome to our presentation"]
    ],
    [
      0.46,
      ["gesturing", "talking"],
      ["person", "woman"],
      ["indoor", "office"],
      ["Today we'll be discussing the results"]
    ],
    // More annotations...
  ]
}
```

## Models

The system uses four types of models:

1. **CLIP** (Contrastive Language-Image Pretraining): Used for agent detection and scene classification
   - Default: `openai/clip-vit-base-patch32`
   - Alternatives: `openai/clip-vit-large-patch14`

2. **ViT** (Vision Transformer): Used for scene classification
   - Default: `google/vit-base-patch16-224`
   - Alternatives: `google/vit-large-patch16-224`

3. **SlowFast**: Used for action recognition
   - Default: `slowfast_r50` (from PyTorchVideo)
   - Alternatives: `slowfast_r101`

4. **Whisper**: Used for speech recognition and transcript generation
   - Default: `openai/whisper-base`
   - Alternatives: `openai/whisper-small`, `openai/whisper-medium`, `openai/whisper-large`

You can use a single model for all annotation categories with the `--unified-model` option, which is useful for comparing model performance across different annotation tasks.

## Validation Metrics

The system includes validation metrics to evaluate annotation quality:

```bash
python validation_metrics.py --input path/to/annotations.json --output-dir path/to/metrics --plot
```

## Performance Benchmarks

To benchmark system performance:

```bash
python run_tests.py --benchmark
```

## Testing

Run all tests:

```bash
python run_tests.py --test all
```

Run specific test categories:

```bash
python run_tests.py --test video
python run_tests.py --test models
python run_tests.py --test annotations
python run_tests.py --test transcript
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

