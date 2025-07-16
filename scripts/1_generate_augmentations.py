#!/usr/bin/env python3
"""
Generate Augmentations Script
============================

Main script to generate augmented samples from Vietnamese SLU dataset.

Usage:
    python scripts/1_generate_augmentations.py --input data/input/dataset.jsonl
    python scripts/1_generate_augmentations.py --input data/input/dataset.jsonl --test-mode
"""

import json
import csv
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from augmenter import VietnameseSLUAugmenter

def load_dataset(file_path: str) -> list:
    """
    Load dataset from JSONL file
    """
    data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    
                    # Validate required fields
                    if 'sentence' not in item or 'intent' not in item:
                        print(f"Warning: Line {line_num} missing required fields, skipping")
                        continue
                    
                    # Ensure entities field exists
                    if 'entities' not in item:
                        item['entities'] = []
                    
                    data.append(item)
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Line {line_num} has invalid JSON, skipping: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"Error: Input file not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(data)} samples from {file_path}")
    return data

def save_augmented_data(augmentations: list, output_dir: str, timestamp: str) -> str:
    """
    Save augmented data to JSON file
    """
    os.makedirs(f"{output_dir}/augmented", exist_ok=True)
    
    output_file = f"{output_dir}/augmented/augmented_data_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'timestamp': timestamp,
                'total_augmentations': len(augmentations),
                'generation_timestamp': datetime.now().isoformat()
            },
            'augmentations': augmentations
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Saved augmented data: {output_file}")
    return output_file

def create_review_csv(augmentations: list, output_dir: str, timestamp: str, max_samples: int = 100) -> str:
    """
    Create CSV file for human review
    """
    os.makedirs(f"{output_dir}/review", exist_ok=True)
    
    review_file = f"{output_dir}/review/review_samples_{timestamp}.csv"
    
    # Take first N samples for review (manageable amount)
    review_samples = augmentations[:max_samples]
    
    with open(review_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Sample_ID',
            'Original_Sentence',
            'Original_Intent', 
            'Original_Entities',
            'Augmented_Sentence',
            'Augmented_Intent',
            'Augmented_Entities',
            'Augmentation_Method',
            'Quality_Score_1_5',
            'Naturalness_Score_1_5',
            'Approved_Yes_No'
        ])
        
        # Data rows
        for i, aug in enumerate(review_samples, 1):
            writer.writerow([
                i,
                aug['original_sentence'],
                aug['original_intent'],
                json.dumps(aug['original_entities'], ensure_ascii=False),
                aug['augmented_sentence'],
                aug['augmented_intent'],
                json.dumps(aug['augmented_entities'], ensure_ascii=False),
                aug['method'],
                '',  # Quality score (to be filled by human)
                '',  # Naturalness score (to be filled by human)
                '',  # Comments (to be filled by human)
                ''   # Approved (to be filled by human)
            ])
    
    print(f"Created review file: {review_file}")
    print(f"   Contains {len(review_samples)} samples for review")
    
    if len(augmentations) > max_samples:
        print(f"   Note: Only first {max_samples} samples included for review")
        print(f"   Total generated: {len(augmentations)} samples")
    
    return review_file

def save_generation_stats(augmenter: VietnameseSLUAugmenter, analysis: dict, 
                         output_dir: str, timestamp: str) -> str:
    """
    Save generation statistics and analysis
    """
    os.makedirs(f"{output_dir}/augmented", exist_ok=True)
    
    stats_file = f"{output_dir}/augmented/generation_stats_{timestamp}.json"
    
    stats = {
        'dataset_analysis': analysis,
        'generation_statistics': augmenter.get_statistics(),
        'quality_issues': augmenter.quality_issues,
        'timestamp': timestamp
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"Saved statistics: {stats_file}")
    return stats_file

def print_summary(analysis: dict, augmentations: list, augmenter: VietnameseSLUAugmenter):
    """
    Print generation summary
    """
    stats = augmenter.get_statistics()
    
    print("\n" + "="*60)
    print("AUGMENTATION GENERATION COMPLETE!")
    print("="*60)
    
    print(f"Dataset Analysis:")
    print(f"   Original samples: {analysis['total_samples']:,}")
    print(f"   Unique intents: {analysis['unique_intents']}")
    print(f"   Unique entity types: {analysis['unique_entities']}")
    print(f"   Rare intents: {len(analysis['rare_intents'])}")
    print(f"   Rare entities: {len(analysis['rare_entities'])}")
    
    print(f"\nGeneration Results:")
    print(f"   Generated samples: {len(augmentations):,}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Quality issues: {stats['quality_issue_count']}")
    
    print(f"\nMethod Breakdown:")
    for method, count in stats['method_breakdown'].items():
        print(f"   {method}: {count}")
    
    print(f"\nNext Steps:")
    print(f"   1. Review the CSV file to approve/reject samples")
    print(f"   2. Fill in Quality_Score_1_5 (1-5) and Naturalness_Score_1_5 (1-5)")
    print(f"   3. Set Approved_Yes_No to 'yes' or 'no'")
    print(f"   4. Run integration script: scripts/3_integrate_approved_data.py")

def main():
    parser = argparse.ArgumentParser(description='Generate Vietnamese SLU augmentations')
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input JSONL file path')
    parser.add_argument('--config', '-c', default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--output', '-o', default='data/output',
                       help='Output directory')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode: process only first 50 samples')
    parser.add_argument('--max-review-samples', type=int, default=100,
                       help='Maximum samples to include in review CSV')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Validate config file
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        print(f"   Please ensure config.yaml exists or specify with --config")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("Vietnamese SLU Data Augmentation")
    print("="*50)
    
    # Load dataset
    print("Loading dataset...")
    data = load_dataset(args.input)
    
    if args.test_mode:
        print("Test mode: Using first 50 samples")
        data = data[:50]
    
    # Initialize augmenter
    print("Initializing augmenter...")
    augmenter = VietnameseSLUAugmenter(args.config)
    
    # Analyze dataset
    print("Analyzing dataset...")
    analysis = augmenter.analyze_dataset(data)
    
    # Generate augmentations
    print("Generating augmentations...")
    
    all_augmentations = []
    rare_intents = set(analysis['rare_intents'])
    rare_entities = set(analysis['rare_entities'])
    
    # Progress bar
    with tqdm(total=len(data), desc="Processing samples") as pbar:
        for item in data:
            augmenter.processed_samples += 1
            
            # Check if sample should be augmented
            if augmenter.should_augment_sample(item, rare_intents, rare_entities):
                sentence = item['sentence']
                intent = item['intent']
                entities = item['entities']
                
                # Generate augmentations for this sample
                augmentations = augmenter.generate_augmentations(sentence, intent, entities)
                all_augmentations.extend(augmentations)
            
            pbar.update(1)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nSaving results...")
    
    # Save augmented data
    data_file = save_augmented_data(all_augmentations, args.output, timestamp)
    
    # Create review CSV
    review_file = create_review_csv(all_augmentations, args.output, timestamp, 
                                   args.max_review_samples)
    
    # Save statistics
    stats_file = save_generation_stats(augmenter, analysis, args.output, timestamp)
    
    # Print summary
    print_summary(analysis, all_augmentations, augmenter)
    
    print(f"\nOutput Files:")
    print(f"   Data: {data_file}")
    print(f"   Review: {review_file}")
    print(f"   Stats: {stats_file}")

if __name__ == "__main__":
    main()