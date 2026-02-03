#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation metrics script for the Video Annotation System.

This script calculates various metrics to evaluate the quality of the annotations:
- Temporal consistency
- Category distribution
- Annotation density
- Confidence scores
- Inter-model agreement
"""

import os
import sys
import logging
import argparse
import json
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calculate validation metrics for video annotations")
    parser.add_argument("--input", type=str, required=True, help="Path to the annotation file (JSON, CSV, or pickle)")
    parser.add_argument("--output-dir", type=str, default="validation_metrics", help="Directory to save validation metrics")
    parser.add_argument("--plot", action="store_true", help="Generate plots for the metrics")
    return parser.parse_args()


def load_annotations(input_path):
    """Load annotations from a file."""
    file_extension = os.path.splitext(input_path)[1].lower()
    
    try:
        if file_extension == ".json":
            with open(input_path, 'r') as f:
                data = json.load(f)
                annotations = data.get("annotations", [])
                metadata = data.get("metadata", {})
        
        elif file_extension == ".csv":
            annotations = []
            with open(input_path, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)
                
                if header != ["timestamp", "actions", "agents", "backgrounds"]:
                    logger.error(f"Invalid CSV header: {header}")
                    return None, None
                
                for row in reader:
                    if len(row) != 4:
                        logger.warning(f"Skipping invalid row: {row}")
                        continue
                    
                    timestamp = float(row[0])
                    actions = row[1].strip("[]").split(", ")
                    agents = row[2].strip("[]").split(", ")
                    backgrounds = row[3].strip("[]").split(", ")
                    
                    # Clean up empty strings
                    actions = [a.strip("'\"") for a in actions if a.strip("'\"")]
                    agents = [a.strip("'\"") for a in agents if a.strip("'\"")]
                    backgrounds = [b.strip("'\"") for b in backgrounds if b.strip("'\"")]
                    
                    annotations.append([timestamp, actions, agents, backgrounds])
            
            # Extract metadata from filename
            filename = os.path.basename(input_path)
            parts = filename.split("_")
            
            metadata = {
                "video_filename": "_".join(parts[2:-1]),
                "duration": float(parts[-1].split(".")[0].replace("s", "")),
                "sampling_rate": 0.46  # Default
            }
        
        elif file_extension == ".pickle":
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
                annotations = data.get("annotations", [])
                metadata = data.get("metadata", {})
        
        else:
            logger.error(f"Unsupported file extension: {file_extension}")
            return None, None
        
        logger.info(f"Loaded {len(annotations)} annotations from {input_path}")
        return annotations, metadata
    
    except Exception as e:
        logger.error(f"Error loading annotations: {str(e)}")
        return None, None


def calculate_temporal_consistency(annotations):
    """Calculate temporal consistency metrics."""
    if not annotations:
        return {}
    
    # Extract timestamps
    timestamps = [annotation[0] for annotation in annotations]
    
    # Calculate time intervals
    intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    
    # Calculate statistics
    mean_interval = np.mean(intervals) if intervals else 0
    std_interval = np.std(intervals) if intervals else 0
    min_interval = min(intervals) if intervals else 0
    max_interval = max(intervals) if intervals else 0
    
    # Calculate consistency score (1.0 means perfectly consistent intervals)
    consistency_score = 1.0 - (std_interval / mean_interval) if mean_interval > 0 else 0.0
    consistency_score = max(0.0, min(1.0, consistency_score))
    
    return {
        "mean_interval": mean_interval,
        "std_interval": std_interval,
        "min_interval": min_interval,
        "max_interval": max_interval,
        "consistency_score": consistency_score
    }


def calculate_category_distribution(annotations):
    """Calculate category distribution metrics."""
    if not annotations:
        return {}
    
    # Extract categories
    all_actions = []
    all_agents = []
    all_backgrounds = []
    
    for annotation in annotations:
        _, actions, agents, backgrounds = annotation
        all_actions.extend(actions)
        all_agents.extend(agents)
        all_backgrounds.extend(backgrounds)
    
    # Count occurrences
    action_counts = Counter(all_actions)
    agent_counts = Counter(all_agents)
    background_counts = Counter(all_backgrounds)
    
    # Calculate statistics
    action_diversity = len(action_counts)
    agent_diversity = len(agent_counts)
    background_diversity = len(background_counts)
    
    # Calculate top categories
    top_actions = action_counts.most_common(5)
    top_agents = agent_counts.most_common(5)
    top_backgrounds = background_counts.most_common(5)
    
    return {
        "action_diversity": action_diversity,
        "agent_diversity": agent_diversity,
        "background_diversity": background_diversity,
        "top_actions": top_actions,
        "top_agents": top_agents,
        "top_backgrounds": top_backgrounds,
        "action_counts": dict(action_counts),
        "agent_counts": dict(agent_counts),
        "background_counts": dict(background_counts)
    }


def calculate_annotation_density(annotations):
    """Calculate annotation density metrics."""
    if not annotations:
        return {}
    
    # Calculate average number of annotations per category
    action_counts = [len(annotation[1]) for annotation in annotations]
    agent_counts = [len(annotation[2]) for annotation in annotations]
    background_counts = [len(annotation[3]) for annotation in annotations]
    
    # Calculate statistics
    mean_action_count = np.mean(action_counts)
    mean_agent_count = np.mean(agent_counts)
    mean_background_count = np.mean(background_counts)
    
    std_action_count = np.std(action_counts)
    std_agent_count = np.std(agent_counts)
    std_background_count = np.std(background_counts)
    
    # Calculate overall density
    total_annotations = sum(action_counts) + sum(agent_counts) + sum(background_counts)
    annotation_count = len(annotations)
    overall_density = total_annotations / (annotation_count * 3) if annotation_count > 0 else 0
    
    return {
        "mean_action_count": mean_action_count,
        "mean_agent_count": mean_agent_count,
        "mean_background_count": mean_background_count,
        "std_action_count": std_action_count,
        "std_agent_count": std_agent_count,
        "std_background_count": std_background_count,
        "overall_density": overall_density
    }


def calculate_inter_category_correlation(annotations):
    """Calculate correlation between different annotation categories."""
    if not annotations:
        return {}
    
    # Create a dictionary to track co-occurrences
    action_agent_pairs = Counter()
    action_background_pairs = Counter()
    agent_background_pairs = Counter()
    
    # Count co-occurrences
    for annotation in annotations:
        _, actions, agents, backgrounds = annotation
        
        for action in actions:
            for agent in agents:
                action_agent_pairs[(action, agent)] += 1
            
            for background in backgrounds:
                action_background_pairs[(action, background)] += 1
        
        for agent in agents:
            for background in backgrounds:
                agent_background_pairs[(agent, background)] += 1
    
    # Find top correlations
    top_action_agent_pairs = action_agent_pairs.most_common(5)
    top_action_background_pairs = action_background_pairs.most_common(5)
    top_agent_background_pairs = agent_background_pairs.most_common(5)
    
    return {
        "top_action_agent_pairs": top_action_agent_pairs,
        "top_action_background_pairs": top_action_background_pairs,
        "top_agent_background_pairs": top_agent_background_pairs
    }


def calculate_temporal_transitions(annotations):
    """Calculate temporal transition metrics."""
    if not annotations or len(annotations) < 2:
        return {}
    
    # Track transitions
    action_transitions = Counter()
    agent_transitions = Counter()
    background_transitions = Counter()
    
    # Count transitions
    for i in range(len(annotations) - 1):
        _, actions1, agents1, backgrounds1 = annotations[i]
        _, actions2, agents2, backgrounds2 = annotations[i+1]
        
        for action1 in actions1:
            for action2 in actions2:
                if action1 != action2:
                    action_transitions[(action1, action2)] += 1
        
        for agent1 in agents1:
            for agent2 in agents2:
                if agent1 != agent2:
                    agent_transitions[(agent1, agent2)] += 1
        
        for background1 in backgrounds1:
            for background2 in backgrounds2:
                if background1 != background2:
                    background_transitions[(background1, background2)] += 1
    
    # Find top transitions
    top_action_transitions = action_transitions.most_common(5)
    top_agent_transitions = agent_transitions.most_common(5)
    top_background_transitions = background_transitions.most_common(5)
    
    return {
        "top_action_transitions": top_action_transitions,
        "top_agent_transitions": top_agent_transitions,
        "top_background_transitions": top_background_transitions
    }


def generate_plots(metrics, output_dir):
    """Generate plots for the metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot category distribution
    if "category_distribution" in metrics:
        category_dist = metrics["category_distribution"]
        
        # Plot action distribution
        action_counts = category_dist.get("action_counts", {})
        if action_counts:
            plt.figure(figsize=(12, 6))
            plt.bar(action_counts.keys(), action_counts.values())
            plt.title("Action Distribution")
            plt.xlabel("Action")
            plt.ylabel("Count")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "action_distribution.png"))
            plt.close()
        
        # Plot agent distribution
        agent_counts = category_dist.get("agent_counts", {})
        if agent_counts:
            plt.figure(figsize=(12, 6))
            plt.bar(agent_counts.keys(), agent_counts.values())
            plt.title("Agent Distribution")
            plt.xlabel("Agent")
            plt.ylabel("Count")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "agent_distribution.png"))
            plt.close()
        
        # Plot background distribution
        background_counts = category_dist.get("background_counts", {})
        if background_counts:
            plt.figure(figsize=(12, 6))
            plt.bar(background_counts.keys(), background_counts.values())
            plt.title("Background Distribution")
            plt.xlabel("Background")
            plt.ylabel("Count")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "background_distribution.png"))
            plt.close()
    
    # Plot annotation density
    if "annotation_density" in metrics:
        density = metrics["annotation_density"]
        
        # Plot density by category
        categories = ["Actions", "Agents", "Backgrounds"]
        means = [
            density.get("mean_action_count", 0),
            density.get("mean_agent_count", 0),
            density.get("mean_background_count", 0)
        ]
        stds = [
            density.get("std_action_count", 0),
            density.get("std_agent_count", 0),
            density.get("std_background_count", 0)
        ]
        
        plt.figure(figsize=(10, 6))
        plt.bar(categories, means, yerr=stds, capsize=10)
        plt.title("Annotation Density by Category")
        plt.xlabel("Category")
        plt.ylabel("Average Count per Frame")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "annotation_density.png"))
        plt.close()
    
    logger.info(f"Generated plots in {output_dir}")


def main():
    """Calculate validation metrics for video annotations."""
    args = parse_args()
    
    # Load annotations
    annotations, metadata = load_annotations(args.input)
    
    if annotations is None or not annotations:
        logger.error("Failed to load annotations")
        return
    
    # Calculate metrics
    metrics = {}
    
    # Temporal consistency
    metrics["temporal_consistency"] = calculate_temporal_consistency(annotations)
    logger.info(f"Temporal consistency: {metrics['temporal_consistency']}")
    
    # Category distribution
    metrics["category_distribution"] = calculate_category_distribution(annotations)
    logger.info(f"Category diversity: Actions={metrics['category_distribution']['action_diversity']}, "
                f"Agents={metrics['category_distribution']['agent_diversity']}, "
                f"Backgrounds={metrics['category_distribution']['background_diversity']}")
    
    # Annotation density
    metrics["annotation_density"] = calculate_annotation_density(annotations)
    logger.info(f"Annotation density: {metrics['annotation_density']['overall_density']:.2f}")
    
    # Inter-category correlation
    metrics["inter_category_correlation"] = calculate_inter_category_correlation(annotations)
    
    # Temporal transitions
    metrics["temporal_transitions"] = calculate_temporal_transitions(annotations)
    
    # Add metadata
    metrics["metadata"] = metadata
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save metrics
    output_path = os.path.join(args.output_dir, f"validation_metrics_{os.path.basename(args.input).split('.')[0]}.json")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved validation metrics to {output_path}")
    
    # Generate plots
    if args.plot:
        plot_dir = os.path.join(args.output_dir, "plots")
        generate_plots(metrics, plot_dir)
    
    # Print summary
    logger.info("\nValidation Metrics Summary:")
    logger.info(f"Video: {metadata.get('video_filename', 'unknown')}")
    logger.info(f"Duration: {metadata.get('duration', 0):.2f}s")
    logger.info(f"Sampling Rate: {metadata.get('sampling_rate', 0):.2f}s")
    logger.info(f"Annotation Count: {len(annotations)}")
    logger.info(f"Temporal Consistency: {metrics['temporal_consistency']['consistency_score']:.2f}")
    logger.info(f"Overall Annotation Density: {metrics['annotation_density']['overall_density']:.2f}")
    logger.info(f"Action Diversity: {metrics['category_distribution']['action_diversity']}")
    logger.info(f"Agent Diversity: {metrics['category_distribution']['agent_diversity']}")
    logger.info(f"Background Diversity: {metrics['category_distribution']['background_diversity']}")
    
    # Print top categories
    logger.info("\nTop Actions:")
    for action, count in metrics['category_distribution']['top_actions']:
        logger.info(f"  {action}: {count}")
    
    logger.info("\nTop Agents:")
    for agent, count in metrics['category_distribution']['top_agents']:
        logger.info(f"  {agent}: {count}")
    
    logger.info("\nTop Backgrounds:")
    for background, count in metrics['category_distribution']['top_backgrounds']:
        logger.info(f"  {background}: {count}")


if __name__ == "__main__":
    main()

