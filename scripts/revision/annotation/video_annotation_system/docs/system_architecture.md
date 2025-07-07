# System Architecture

This document provides a detailed overview of the Multi-Category Video Annotation System architecture, explaining the design decisions, components, and data flow.

## Table of Contents

- [Overview](#overview)
- [Design Principles](#design-principles)
- [System Components](#system-components)
  - [Video Processing Module](#video-processing-module)
  - [Model Integration Module](#model-integration-module)
  - [Annotation Generation Module](#annotation-generation-module)
  - [Output Formatting Module](#output-formatting-module)
  - [Command-Line Interface](#command-line-interface)
  - [Testing Framework](#testing-framework)
  - [Validation Metrics Module](#validation-metrics-module)
- [Data Flow](#data-flow)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Performance Considerations](#performance-considerations)
- [Future Extensions](#future-extensions)

## Overview

The Multi-Category Video Annotation System is designed to analyze videos and produce structured annotations at precise 0.46-second intervals using state-of-the-art AI models. The system follows a modular architecture that separates concerns and allows for easy extension and maintenance.

## Design Principles

The system architecture is guided by the following principles:

1. **Modularity**: The system is divided into independent modules with clear responsibilities.
2. **Extensibility**: New models and annotation categories can be added without modifying the core system.
3. **Reliability**: Comprehensive testing ensures the system works correctly in various scenarios.
4. **Performance**: Batch processing and GPU acceleration optimize performance for large videos.
5. **Usability**: A simple command-line interface makes the system accessible to researchers.

## System Components

### Video Processing Module

**Purpose**: Handle video loading, validation, and frame extraction.

**Key Functions**:
- Validate video files
- Extract video metadata (duration, fps, resolution)
- Extract frames at precise 0.46-second intervals
- Preprocess frames for model input

**Implementation**: `utils/video_utils.py`

The video processing module uses OpenCV for video handling and frame extraction. It ensures that frames are extracted at precise 0.46-second intervals, which is critical for aligning annotations with fMRI data. The module also handles various video formats and provides utility functions for video metadata extraction.

### Model Integration Module

**Purpose**: Load and manage AI models for annotation generation.

**Key Functions**:
- Load CLIP model for agent and scene detection
- Load ViT model for scene detection
- Load SlowFast model for action recognition
- Provide a unified interface for model inference

**Implementation**: `models/model_loader.py`

The model integration module uses the Hugging Face Transformers library to load and manage AI models. It provides a unified interface for model inference, abstracting away the differences between model architectures. The module also handles model caching to avoid reloading models for each video.

### Annotation Generation Module

**Purpose**: Generate annotations from video frames using AI models.

**Key Functions**:
- Generate action annotations
- Generate agent annotations
- Generate scene/background annotations
- Combine annotations from different models

**Implementation**: `models/annotation_generator.py`

The annotation generation module uses the loaded models to generate annotations for each frame. It processes frames in batches for efficiency and combines annotations from different models into a unified format. The module also handles model-specific preprocessing and postprocessing.

### Output Formatting Module

**Purpose**: Format and save annotations in various formats.

**Key Functions**:
- Format annotations in a structured format
- Save annotations in JSON format
- Save annotations in CSV format
- Save annotations in pickle format

**Implementation**: `utils/annotation_utils.py`

The output formatting module takes the raw annotations from the annotation generation module and formats them in a structured format. It then saves the annotations in the specified format (JSON, CSV, or pickle). The module also adds metadata to the annotations, such as video duration, fps, and resolution.

### Command-Line Interface

**Purpose**: Provide a user-friendly interface for system operation.

**Key Functions**:
- Parse command-line arguments
- Handle input validation
- Coordinate the execution of other modules
- Report progress and results

**Implementation**: `video_annotator.py`

The command-line interface is the main entry point for the system. It parses command-line arguments, validates inputs, and coordinates the execution of other modules. It also handles error reporting and progress updates.

### Testing Framework

**Purpose**: Ensure system reliability and performance.

**Key Functions**:
- Run unit tests for individual modules
- Run integration tests for end-to-end processing
- Measure system performance
- Validate system output

**Implementation**: `tests/` directory, `run_tests.py`

The testing framework includes unit tests for individual modules, integration tests for end-to-end processing, and performance benchmarks. It ensures that the system works correctly in various scenarios and helps identify performance bottlenecks.

### Validation Metrics Module

**Purpose**: Evaluate annotation quality and consistency.

**Key Functions**:
- Calculate temporal consistency metrics
- Analyze category distribution
- Measure annotation density
- Identify correlations between categories
- Generate visualizations

**Implementation**: `validation_metrics.py`

The validation metrics module calculates various metrics to evaluate the quality and consistency of the annotations. It helps researchers assess the reliability of the annotations and identify potential issues.

## Data Flow

The data flows through the system as follows:

1. **Input**: Video file or folder of videos
2. **Video Processing**: Extract frames at 0.46-second intervals
3. **Model Integration**: Load AI models
4. **Annotation Generation**: Generate annotations for each frame
5. **Output Formatting**: Format and save annotations
6. **Validation**: Calculate metrics to evaluate annotation quality

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Video Input    │────▶│  Frame          │────▶│  Model          │
│  - Single file  │     │  Extraction     │     │  Integration    │
│  - Folder       │     │  (0.46s)        │     │  (CLIP, ViT,    │
└─────────────────┘     └─────────────────┘     │   SlowFast)     │
                                                └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐     ┌───────▼────────┐
│  Output         │◀────│  Annotation     │◀────│  Annotation    │
│  - JSON         │     │  Formatting     │     │  Generation    │
│  - CSV          │     │  - Actions      │     │  - Actions     │
│  - Pickle       │     │  - Agents       │     │  - Agents      │
└─────────────────┘     │  - Backgrounds  │     │  - Backgrounds │
                        └─────────────────┘     └─────────────────┘
```

## File Structure

The system is organized into the following file structure:

```
video_annotation_system/
├── video_annotator.py          # Main script
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
├── README.md                   # Documentation
├── run_tests.py                # Test runner
├── validation_metrics.py       # Validation metrics
├── docs/                       # Documentation
│   ├── usage_examples.md       # Usage examples
│   └── system_architecture.md  # System architecture
├── models/                     # Model integration
│   ├── __init__.py
│   ├── model_loader.py         # Model loading
│   └── annotation_generator.py # Annotation generation
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── video_utils.py          # Video processing
│   └── annotation_utils.py     # Annotation formatting
└── tests/                      # Tests
    ├── __init__.py
    ├── test_basic.py           # Basic tests
    ├── test_video_processing.py # Video processing tests
    ├── test_model_loading.py   # Model loading tests
    ├── test_annotation_generation.py # Annotation generation tests
    ├── test_annotation_formatting.py # Annotation formatting tests
    └── test_command_line.py    # Command-line interface tests
```

## Dependencies

The system relies on the following key dependencies:

- **OpenCV**: Video processing and frame extraction
- **PyTorch**: Deep learning framework for model inference
- **Hugging Face Transformers**: Loading and using pre-trained models
- **NumPy**: Numerical computing
- **Matplotlib**: Visualization for validation metrics
- **tqdm**: Progress bars for long-running operations

## Performance Considerations

The system is designed to handle large videos efficiently:

- **Batch Processing**: Frames are processed in batches to maximize GPU utilization.
- **GPU Acceleration**: The system can use CUDA-compatible GPUs for faster processing.
- **Memory Management**: The system minimizes memory usage by processing frames sequentially.
- **Caching**: Models are loaded once and reused for multiple videos.

## Future Extensions

The system is designed to be easily extended in the following ways:

- **New Models**: Additional models can be integrated by adding new functions to the model loader.
- **New Annotation Categories**: New categories can be added by extending the annotation generator.
- **New Output Formats**: Additional output formats can be supported by adding new functions to the output formatter.
- **Web Interface**: A web interface could be added to make the system more accessible.
- **Real-time Processing**: The system could be extended to process video streams in real-time.
- **Distributed Processing**: The system could be extended to distribute processing across multiple machines.

