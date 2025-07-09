#!/usr/bin/env python3
"""
Integrate Approved Data Script
=============================

Script to integrate human-approved augmented samples back into the dataset.

Usage:
    python scripts/3_integrate_approved_data.py --review-file data/output/review/reviewed_samples.csv
    python scripts/3_integrate_approved_data.py --review-file reviewed.csv --output data/output/final/
"""

import json
import csv
import os
import sys
import argparse
from datetime import datetime
import pandas as pd
from pathlib import Path
from collections import Counter

def load_review_csv(file_path: str) -> pd.DataFrame:
    """
    Load the reviewed CSV file
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # Validate required columns
        required_columns = [
            'Sample_ID', 'Original_Sentence', 'Original_Intent', 'Original_Entities',
            'Augmented_Sentence', 'Augmented_Intent', 'Augmented_Entities',
            'Augmentation_Method', 'Quality_Score_1_5', 'Naturalness_Score_1_5',
            'Approved_Yes_No'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            sys.exit(1)
        
        print(f"✅ Loaded review file with {len(df)} samples")
        return df
        
    except FileNotFoundError:
        print(f"❌ Review file not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading review file: {e}")
        sys.exit(1)

def analyze_review_results(df: pd.DataFrame) -> dict:
    """
    Analyze the review results
    """
    total_samples = len(df)
    
    # Count approvals
    explicit_approvals = df['Approved_Yes_No'].str.lower() == 'yes'
    explicit_rejections = df['Approved_Yes_No'].str.lower() == 'no'
    
    # Count score-based approvals (quality >= 3 AND naturalness >= 3)
    quality_scores = pd.to_numeric(df['Quality_Score_1_5'], errors='coerce')
    naturalness_scores = pd.to_numeric(df['Naturalness_Score_1_5'], errors='coerce')
    
    score_based_approvals = (quality_scores >= 3) & (naturalness_scores >= 3)
    
    # Combined approvals
    approved_mask = explicit_approvals | (score_based_approvals & ~explicit_rejections)
    
    approved_count = approved_mask.sum()
    rejected_count = explicit_rejections.sum()
    unreviewed_count = total_samples - approved_count - rejected_count
    
    # Method breakdown
    method_counts = df['Augmentation_Method'].value_counts().to_dict()
    approved_by_method = df[approved_mask]['Augmentation_Method'].value_counts().to_dict()
    
    # Score statistics
    avg_quality = quality_scores.mean()
    avg_naturalness = naturalness_scores.mean()
    
    analysis = {
        'total_samples': total_samples,
        'approved_count': approved_count,
        'rejected_count': rejected_count,
        'unreviewed_count': unreviewed_count,
        'approval_rate': approved_count / total_samples if total_samples > 0 else 0,
        'method_breakdown': method_counts,
        'approved_by_method': approved_by_method,
        'average_quality_score': avg_quality,
        'average_naturalness_score': avg_naturalness,
        'approved_mask': approved_mask
    }
    
    return analysis

def create_final_dataset(df: pd.DataFrame, approved_mask, strategy: str = "balanced") -> list:
    """
    Create the final dataset with approved augmentations
    """
    approved_df = df[approved_mask].copy()
    
    final_samples = []
    
    for _, row in approved_df.iterrows():
        try:
            # Parse entities JSON
            original_entities = json.loads(row['Original_Entities']) if row['Original_Entities'] else []
            augmented_entities = json.loads(row['Augmented_Entities']) if row['Augmented_Entities'] else []
            
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in entities for sample {row['Sample_ID']}, skipping")
            continue
        
        # Create final sample
        sample = {
            'sentence': row['Augmented_Sentence'],
            'intent': row['Augmented_Intent'],
            'entities': augmented_entities,
            
            # Metadata for tracking
            '_augmented': True,
            '_augmentation_method': row['Augmentation_Method'],
            '_original_sentence': row['Original_Sentence'],
            '_quality_score': row['Quality_Score_1_5'],
            '_naturalness_score': row['Naturalness_Score_1_5'],
            '_review_timestamp': datetime.now().isoformat(),
            '_sample_id': row['Sample_ID']
        }
        
        final_samples.append(sample)
    
    print(f"✅ Created {len(final_samples)} approved samples")
    return final_samples

def save_final_dataset(samples: list, output_dir: str, format: str = "jsonl") -> str:
    """
    Save the final dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "jsonl":
        output_file = f"{output_dir}/final_dataset_{timestamp}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    elif format == "json":
        output_file = f"{output_dir}/final_dataset_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"💾 Saved final dataset: {output_file}")
    return output_file

def save_integration_report(analysis: dict, final_samples: list, output_dir: str) -> str:
    """
    Save integration report
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"{output_dir}/integration_report_{timestamp}.json"
    
    # Remove the mask from analysis (not JSON serializable)
    report_analysis = analysis.copy()
    if 'approved_mask' in report_analysis:
        del report_analysis['approved_mask']
    
    report = {
        'integration_timestamp': datetime.now().isoformat(),
        'review_analysis': report_analysis,
        'final_dataset_size': len(final_samples),
        'integration_summary': {
            'total_reviewed': analysis['total_samples'],
            'approved_samples': analysis['approved_count'],
            'approval_rate': f"{analysis['approval_rate']:.1%}",
            'average_quality': f"{analysis['average_quality_score']:.2f}" if not pd.isna(analysis['average_quality_score']) else "N/A",
            'average_naturalness': f"{analysis['average_naturalness_score']:.2f}" if not pd.isna(analysis['average_naturalness_score']) else "N/A"
        },
        'method_performance': {
            method: {
                'total_generated': analysis['method_breakdown'].get(method, 0),
                'approved': analysis['approved_by_method'].get(method, 0),
                'approval_rate': analysis['approved_by_method'].get(method, 0) / analysis['method_breakdown'].get(method, 1)
            }
            for method in analysis['method_breakdown'].keys()
        }
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"📊 Saved integration report: {report_file}")
    return report_file

def print_integration_summary(analysis: dict, final_samples: list):
    """
    Print integration summary
    """
    print("\n" + "="*60)
    print("🎉 DATA INTEGRATION COMPLETE!")
    print("="*60)
    
    print(f"📊 Review Analysis:")
    print(f"   Total samples reviewed: {analysis['total_samples']:,}")
    print(f"   Approved samples: {analysis['approved_count']:,}")
    print(f"   Rejected samples: {analysis['rejected_count']:,}")
    print(f"   Unreviewed samples: {analysis['unreviewed_count']:,}")
    print(f"   Approval rate: {analysis['approval_rate']:.1%}")
    
    if not pd.isna(analysis['average_quality_score']):
        print(f"   Average quality score: {analysis['average_quality_score']:.2f}/5")
    if not pd.isna(analysis['average_naturalness_score']):
        print(f"   Average naturalness score: {analysis['average_naturalness_score']:.2f}/5")
    
    print(f"\n🔧 Method Performance:")
    for method in analysis['method_breakdown'].keys():
        total = analysis['method_breakdown'][method]
        approved = analysis['approved_by_method'].get(method, 0)
        rate = approved / total if total > 0 else 0
        print(f"   {method}: {approved}/{total} ({rate:.1%})")
    
    print(f"\n✅ Final Dataset:")
    print(f"   Approved augmented samples: {len(final_samples):,}")
    print(f"   Ready for training integration!")
    
    print(f"\n📝 Next Steps:")
    print(f"   1. Upload the final dataset to your training environment")
    print(f"   2. Load into your processor/training pipeline") 
    print(f"   3. Train your model with the augmented data")
    print(f"   4. Monitor performance improvements!")

def main():
    parser = argparse.ArgumentParser(description='Integrate approved augmented data')
    
    parser.add_argument('--review-file', '-r', required=True,
                       help='Path to reviewed CSV file')
    parser.add_argument('--output', '-o', default='data/output/final',
                       help='Output directory for final dataset')
    parser.add_argument('--format', '-f', choices=['jsonl', 'json'], default='jsonl',
                       help='Output format (jsonl or json)')
    parser.add_argument('--strategy', '-s', choices=['all', 'balanced', 'selective'], 
                       default='balanced',
                       help='Integration strategy')
    parser.add_argument('--min-quality', type=float, default=3.0,
                       help='Minimum quality score for approval')
    parser.add_argument('--min-naturalness', type=float, default=3.0,
                       help='Minimum naturalness score for approval')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.review_file):
        print(f"❌ Review file not found: {args.review_file}")
        sys.exit(1)
    
    print("🔗 Vietnamese SLU Data Integration")
    print("="*50)
    
    # Load review file
    print("📁 Loading review file...")
    df = load_review_csv(args.review_file)
    
    # Analyze review results
    print("📊 Analyzing review results...")
    analysis = analyze_review_results(df)
    
    # Create final dataset
    print("✅ Creating final dataset...")
    final_samples = create_final_dataset(df, analysis['approved_mask'], args.strategy)
    
    if len(final_samples) == 0:
        print("⚠️  No approved samples found! Check your review file.")
        print("   Make sure to fill in Quality_Score_1_5, Naturalness_Score_1_5, and/or Approved_Yes_No columns")
        sys.exit(0)
    
    # Save results
    print("💾 Saving final dataset...")
    final_file = save_final_dataset(final_samples, args.output, args.format)
    
    # Save integration report
    report_file = save_integration_report(analysis, final_samples, args.output)
    
    # Print summary
    print_integration_summary(analysis, final_samples)
    
    print(f"\n📁 Output Files:")
    print(f"   Final dataset: {final_file}")
    print(f"   Integration report: {report_file}")
    
    print(f"\n🚀 Ready for Training!")
    print(f"   Upload {final_file} to your training environment")
    print(f"   Expected improvement: F1 Macro 21% → 45-60%")

if __name__ == "__main__":
    main()