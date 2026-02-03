# Usage Examples for Multi-Category Video Annotation System

This document provides detailed examples of how to use the Multi-Category Video Annotation System for various scenarios.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Customizing Output Format](#customizing-output-format)
- [Transcript Generation](#transcript-generation)
- [Batch Processing](#batch-processing)
- [Using Custom Models](#using-custom-models)
- [Performance Optimization](#performance-optimization)
- [Validation and Analysis](#validation-and-analysis)
- [Integration with fMRI Analysis](#integration-with-fmri-analysis)
- [Troubleshooting](#troubleshooting)

## Basic Usage

### Annotating a Single Video

To annotate a single video with default settings:

```bash
python video_annotator.py --input path/to/video.mp4 --output-dir path/to/output
```

This command will:
1. Load the video file
2. Extract frames at 0.46-second intervals
3. Generate annotations for actions, agents, and backgrounds
4. Save the annotations in JSON format in the specified output directory

### Viewing the Results

After running the annotation, you can view the results:

```bash
# View the JSON output
cat path/to/output/video_annotations_video_10.50s.json

# Or use Python to load and explore the annotations
python
>>> import json
>>> with open('path/to/output/video_annotations_video_10.50s.json', 'r') as f:
...     data = json.load(f)
>>> print(f"Video duration: {data['metadata']['duration']}s")
>>> print(f"Number of annotations: {len(data['annotations'])}")
>>> print(f"First annotation: {data['annotations'][0]}")
```

## Customizing Output Format

### JSON Format (Default)

```bash
python video_annotator.py --input path/to/video.mp4 --output-dir path/to/output --format json
```

### CSV Format

```bash
python video_annotator.py --input path/to/video.mp4 --output-dir path/to/output --format csv
```

CSV files can be easily opened in spreadsheet applications like Excel or imported into data analysis tools like pandas:

```python
import pandas as pd
df = pd.read_csv('path/to/output/video_annotations_video_10.50s.csv')
print(df.head())
```

### Pickle Format

```bash
python video_annotator.py --input path/to/video.mp4 --output-dir path/to/output --format pickle
```

Pickle files can be loaded in Python for efficient processing:

```python
import pickle
with open('path/to/output/video_annotations_video_10.50s.pickle', 'rb') as f:
    data = pickle.load(f)
print(data['metadata'])
```

## Transcript Generation

### Basic Transcript Generation

To generate transcripts from video audio:

```bash
python video_annotator.py --input path/to/video.mp4 --generate-transcript --output-dir path/to/output
```

This command will:
1. Extract audio from the video
2. Process the audio with the default Whisper model (openai/whisper-base)
3. Generate a transcript with timestamps
4. Align the transcript with the visual annotations
5. Include the transcript in the output annotations

### Customizing Transcript Generation

You can customize the transcript generation process with various options:

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

Options explained:
- `--whisper-model`: Specify which Whisper model to use (base, small, medium, large)
- `--language`: Specify the language code (e.g., en, fr, de, es, etc.)
- `--chunk-size`: Size of audio chunks to process (in seconds)
- `--overlap`: Overlap between chunks (in seconds)
- `--keep-audio`: Keep the extracted audio file (useful for debugging)

### Viewing Transcript Results

The transcript is included in the annotation output:

```python
import json
with open('path/to/output/video_annotations_video_10.50s.json', 'r') as f:
    data = json.load(f)

# Check if transcript is included
has_transcript = data['metadata'].get('has_transcript', False)
print(f"Has transcript: {has_transcript}")

# View the first few annotations with transcripts
if has_transcript:
    for i, annotation in enumerate(data['annotations'][:5]):
        timestamp = annotation[0]
        transcript = annotation[4][0] if len(annotation) > 4 and annotation[4] else "No transcript"
        print(f"[{timestamp:.2f}s] Transcript: {transcript}")
```

### Processing Only Transcript

If you only need the transcript without visual annotations:

```bash
python video_annotator.py --input path/to/video.mp4 \
                         --generate-transcript \
                         --whisper-model openai/whisper-medium \
                         --output-dir path/to/output
```

### Extracting Transcript to a Separate File

You can extract the transcript to a separate file for further processing:

```python
import json

# Load annotations with transcript
with open('path/to/output/video_annotations_video_10.50s.json', 'r') as f:
    data = json.load(f)

# Extract transcript
if data['metadata'].get('has_transcript', False):
    transcript_text = []
    for annotation in data['annotations']:
        if len(annotation) > 4 and annotation[4]:
            timestamp = annotation[0]
            text = annotation[4][0] if annotation[4][0] else ""
            if text:
                transcript_text.append(f"[{timestamp:.2f}s] {text}")
    
    # Save to a text file
    with open('path/to/output/transcript.txt', 'w') as f:
        f.write('\n'.join(transcript_text))
    
    print(f"Transcript saved to path/to/output/transcript.txt")
```

## Batch Processing

### Processing All Videos in a Folder

To process all videos in a folder and its subdirectories:

```bash
python video_annotator.py --input-folder path/to/videos --output-dir path/to/output
```

This will:
1. Recursively find all video files in the specified folder
2. Process each video with the default settings
3. Save the annotations in the output directory with filenames based on the original video names

### Batch Processing with Transcript Generation

To process all videos in a folder and generate transcripts:

```bash
python video_annotator.py --input-folder path/to/videos \
                         --generate-transcript \
                         --whisper-model openai/whisper-base \
                         --output-dir path/to/output
```

### Processing a Specific Set of Videos

You can use shell commands to process a specific set of videos:

```bash
# Process all MP4 files in a folder
for video in /path/to/videos/*.mp4; do
    python video_annotator.py --input "$video" --generate-transcript --output-dir path/to/output
done

# Process videos matching a pattern
for video in /path/to/videos/*lecture*.mp4; do
    python video_annotator.py --input "$video" --generate-transcript --output-dir path/to/output
done
```

## Using Custom Models

### Specifying Different Models for Each Category

You can specify different models for each annotation category:

```bash
python video_annotator.py --input path/to/video.mp4 \
                         --action-model facebook/slowfast \
                         --agent-model openai/clip-vit-base-patch32 \
                         --scene-model google/vit-base-patch16-224 \
                         --output-dir path/to/output
```

### Using a Unified Model

You can use a single model for all annotation categories:

```bash
python video_annotator.py --input path/to/video.mp4 \
                         --unified-model clip \
                         --output-dir path/to/output
```

### Using a Specific Model Variant

You can specify a particular model variant:

```bash
python video_annotator.py --input path/to/video.mp4 \
                         --unified-model clip \
                         --model openai/clip-vit-large-patch14 \
                         --output-dir path/to/output
```

### Using a Specific Whisper Model

You can specify which Whisper model to use for transcript generation:

```bash
python video_annotator.py --input path/to/video.mp4 \
                         --generate-transcript \
                         --whisper-model openai/whisper-large \
                         --output-dir path/to/output
```

### Comparing Different Models

To compare how different models perform on the same video:

```bash
# Using CLIP for all categories
python video_annotator.py --input path/to/video.mp4 \
                         --unified-model clip \
                         --output-dir clip_results

# Using ViT for all categories
python video_annotator.py --input path/to/video.mp4 \
                         --unified-model vit \
                         --output-dir vit_results

# Using SlowFast for all categories
python video_annotator.py --input path/to/video.mp4 \
                         --unified-model slowfast \
                         --output-dir slowfast_results
```

## Performance Optimization

### Using GPU Acceleration

If you have a CUDA-compatible GPU, you can use it for faster processing:

```bash
python video_annotator.py --input path/to/video.mp4 \
                         --device cuda \
                         --output-dir path/to/output
```

### Adjusting Batch Size

You can adjust the batch size for model inference:

```bash
python video_annotator.py --input path/to/video.mp4 \
                         --batch-size 16 \
                         --output-dir path/to/output
```

### Optimizing for Large Videos

For very large videos, you may want to adjust the chunk size for transcript generation:

```bash
python video_annotator.py --input path/to/large_video.mp4 \
                         --generate-transcript \
                         --chunk-size 60 \
                         --overlap 10 \
                         --output-dir path/to/output
```

## Validation and Analysis

### Calculating Validation Metrics

To calculate validation metrics for your annotations:

```bash
python validation_metrics.py --input path/to/output/video_annotations_video_10.50s.json \
                            --output-dir path/to/metrics \
                            --plot
```

### Analyzing Transcript Statistics

To analyze transcript statistics:

```python
import json

# Load annotations with transcript
with open('path/to/output/video_annotations_video_10.50s.json', 'r') as f:
    data = json.load(f)

# Extract transcript segments
if data['metadata'].get('has_transcript', False):
    # Count words
    word_count = 0
    for annotation in data['annotations']:
        if len(annotation) > 4 and annotation[4]:
            text = annotation[4][0] if annotation[4][0] else ""
            words = text.split()
            word_count += len(words)
    
    # Calculate words per minute
    duration_minutes = data['metadata']['duration'] / 60
    words_per_minute = word_count / duration_minutes if duration_minutes > 0 else 0
    
    print(f"Total words: {word_count}")
    print(f"Duration: {duration_minutes:.2f} minutes")
    print(f"Words per minute: {words_per_minute:.2f}")
```

## Integration with fMRI Analysis

### Exporting Annotations for fMRI Analysis

To export annotations in a format suitable for fMRI analysis:

```python
import json
import numpy as np
import pandas as pd

# Load annotations
with open('path/to/output/video_annotations_video_10.50s.json', 'r') as f:
    data = json.load(f)

# Create a DataFrame for fMRI analysis
annotations = data['annotations']
df = pd.DataFrame({
    'timestamp': [ann[0] for ann in annotations],
    'actions': [','.join(ann[1]) for ann in annotations],
    'agents': [','.join(ann[2]) for ann in annotations],
    'backgrounds': [','.join(ann[3]) for ann in annotations]
})

# Add transcript if available
if data['metadata'].get('has_transcript', False):
    df['transcript'] = [','.join(ann[4]) if len(ann) > 4 and ann[4] else "" for ann in annotations]

# Save as CSV for fMRI analysis
df.to_csv('path/to/output/fmri_annotations.csv', index=False)
```

## Troubleshooting

### Handling Video Format Issues

If you encounter issues with video formats:

```bash
# Convert video to a compatible format using ffmpeg
ffmpeg -i input_video.avi -c:v libx264 -c:a aac -strict experimental output_video.mp4

# Then process the converted video
python video_annotator.py --input output_video.mp4 --output-dir path/to/output
```

### Debugging Transcript Generation

If transcript generation is not working as expected:

```bash
# Keep the extracted audio file for inspection
python video_annotator.py --input path/to/video.mp4 \
                         --generate-transcript \
                         --keep-audio \
                         --output-dir path/to/output

# Check the audio file quality
ffplay path/to/output/video_audio.wav

# Try a different Whisper model
python video_annotator.py --input path/to/video.mp4 \
                         --generate-transcript \
                         --whisper-model openai/whisper-medium \
                         --output-dir path/to/output
```

### Memory Issues

If you encounter memory issues with large videos:

```bash
# Reduce batch size
python video_annotator.py --input path/to/large_video.mp4 \
                         --batch-size 4 \
                         --output-dir path/to/output

# Process in smaller chunks
ffmpeg -i large_video.mp4 -ss 00:00:00 -to 00:10:00 -c copy part1.mp4
ffmpeg -i large_video.mp4 -ss 00:10:00 -to 00:20:00 -c copy part2.mp4

python video_annotator.py --input part1.mp4 --output-dir path/to/output
python video_annotator.py --input part2.mp4 --output-dir path/to/output
```

