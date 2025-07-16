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
import numpy as np # <-- THÊM DÒNG NÀY

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
        raise
    except Exception as e:
        print(f"❌ Error loading review file: {e}")
        sys.exit(1)

def analyze_review_results(df: pd.DataFrame) -> dict:
    """
    Analyze the review results and determine approved/rejected samples
    """
    total_samples = len(df)

    # Clean and analyze 'Approved_Yes_No' column
    explicit_approvals = df['Approved_Yes_No'].str.lower().str.strip() == 'yes'
    explicit_rejections = df['Approved_Yes_No'].str.lower().str.strip() == 'no'

    # Convert score columns to numeric, coercing errors to NaN
    quality_scores = pd.to_numeric(df['Quality_Score_1_5'], errors='coerce')
    naturalness_scores = pd.to_numeric(df['Naturalness_Score_1_5'], errors='coerce')

    # Define score-based approval logic (e.g., both scores >= 3)
    # Samples with NaN scores will not be included in score_based_approvals
    score_based_approvals = (quality_scores >= 3) & (naturalness_scores >= 3)

    # Define the final approval mask: either explicitly approved OR (score-based approved AND not explicitly rejected)
    approved_mask = explicit_approvals | (score_based_approvals & ~explicit_rejections)
    
    # Define rejection mask: explicitly rejected OR (not score-based approved AND not explicitly approved)
    rejected_mask = explicit_rejections | (~score_based_approvals & ~explicit_approvals)


    print(f"Debug: Explicit Approvals Count: {explicit_approvals.sum()}")
    print(f"Debug: Explicit Rejections Count: {explicit_rejections.sum()}")
    print(f"Debug: Score-based Approvals Count: {score_based_approvals.sum()}")
    print(f"Debug: Final Approved Mask Count: {approved_mask.sum()}")

    return {
        "total_samples": total_samples,
        "approved_mask": approved_mask, # This is a pandas Series, not directly serializable for JSON report
        "explicit_approvals_count": explicit_approvals.sum(),
        "explicit_rejections_count": explicit_rejections.sum(),
        "score_based_approvals_count": score_based_approvals.sum(),
        "rejected_count": rejected_mask.sum(), # This will be added to report, might be np.int64
        "integration_strategy": "balanced" # Default strategy, can be adjusted if multiple strategies are implemented
    }

def create_final_dataset(df: pd.DataFrame, approved_mask: pd.Series, strategy: str = "balanced") -> list:
    """
    Create the final dataset with approved augmentations
    """
    # Use .loc to ensure we work on a copy to avoid SettingWithCopyWarning
    approved_df = df.loc[approved_mask].copy()

    final_samples = []

    for _, row in approved_df.iterrows():
        try:
            # Ensure entity columns are treated as strings and handle potential NaN or empty strings
            original_entities_str = str(row['Original_Entities']).strip() if pd.notna(row['Original_Entities']) else ''
            augmented_entities_str = str(row['Augmented_Entities']).strip() if pd.notna(row['Augmented_Entities']) else ''

            # Attempt to load JSON, default to empty list if parsing fails or string is empty
            original_entities = json.loads(original_entities_str) if original_entities_str else []
            augmented_entities = json.loads(augmented_entities_str) if augmented_entities_str else []

        except json.JSONDecodeError as e:
            # Use original sample ID for more useful warning
            sample_id = row.get('Sample_ID', 'N/A')
            print(f"Warning: Invalid JSON in entities for sample {sample_id}, skipping. Error: {e}")
            continue # Skip this sample if entities are malformed JSON
        except Exception as e:
            sample_id = row.get('Sample_ID', 'N/A')
            print(f"Warning: Unexpected error processing entities for sample {sample_id}, skipping. Error: {e}")
            continue

        # Ensure entities are always lists, even if JSON was null/empty string or not a list
        if not isinstance(original_entities, list):
            original_entities = []
        if not isinstance(augmented_entities, list):
            augmented_entities = []

        final_samples.append({
            "sample_id": int(row['Sample_ID']) if pd.notna(row['Sample_ID']) else None, # Convert to Python int
            "original_sentence": row['Original_Sentence'],
            "original_intent": row['Original_Intent'],
            "original_entities": original_entities,
            "augmented_sentence": row['Augmented_Sentence'],
            "augmented_intent": row['Augmented_Intent'],
            "augmented_entities": augmented_entities,
            "augmentation_method": row['Augmentation_Method'],
            "quality_score": int(row['Quality_Score_1_5']) if pd.notna(row['Quality_Score_1_5']) else None, # Convert to Python int
            "naturalness_score": int(row['Naturalness_Score_1_5']) if pd.notna(row['Naturalness_Score_1_5']) else None, # Convert to Python int
            "approved": str(row['Approved_Yes_No']).lower().strip() == 'yes'
        })

    return final_samples

def save_final_dataset(final_samples: list, output_dir: str, file_format: str = "jsonl") -> str:
    """
    Save the final dataset to a JSONL or CSV file
    """
    output_path = Path(output_dir) / "final"
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"final_dataset_{timestamp}.{file_format}"
    file_path = output_path / file_name

    if file_format == "jsonl":
        with open(file_path, 'w', encoding='utf-8') as f:
            for sample in final_samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')
    elif file_format == "csv":
        if not final_samples:
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(final_samples)
        df.to_csv(file_path, index=False, encoding='utf-8')
    else:
        raise ValueError(f"Unsupported output format: {file_format}")
        
    return str(file_path)

# Hàm trợ giúp để chuyển đổi các kiểu dữ liệu numpy sang kiểu Python chuẩn để JSON hóa
def convert_numpy_types_for_json(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types_for_json(elem) for elem in obj]
    elif isinstance(obj, np.integer): # Xử lý numpy.int64, numpy.int32, v.v.
        return int(obj) # Chuyển đổi thành int Python tiêu chuẩn
    elif isinstance(obj, np.floating): # Xử lý numpy.float64, numpy.float32, v.v.
        return float(obj) # Chuyển đổi thành float Python tiêu chuẩn
    elif pd.isna(obj): # Xử lý các giá trị NA của pandas (vd: từ scores bị lỗi)
        return None
    else:
        return obj

def save_integration_report(analysis: dict, final_samples: list, output_dir: str) -> str:
    """
    Save the integration report to a JSON file
    """
    output_path = Path(output_dir) / "report"
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"integration_report_{timestamp}.json"
    file_path = output_path / file_name
    
    report = {
        "report_generated_at": datetime.now().isoformat(),
        "total_samples_in_review_file": analysis['total_samples'],
        "approved_samples_count": len(final_samples), # Đây đã là int chuẩn
        "rejected_samples_count": analysis['rejected_count'], 
        "explicitly_approved_count": analysis['explicit_approvals_count'],
        "explicitly_rejected_count": analysis['explicit_rejections_count'],
        "score_based_approvals_count": analysis['score_based_approvals_count'],
        "integration_strategy": analysis['integration_strategy'], 
        "notes": "This report summarizes the integration of human-reviewed augmented samples."
    }

    # Chuyển đổi bất kỳ kiểu numpy nào trong từ điển báo cáo sang kiểu Python chuẩn
    report_to_dump = convert_numpy_types_for_json(report)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(report_to_dump, f, ensure_ascii=False, indent=2)
        
    return str(file_path)

def print_integration_summary(analysis: dict, final_samples: list):
    """
    Print a summary of the integration process
    """
    total_samples = analysis['total_samples']
    approved_count = len(final_samples)
    rejected_count = analysis['rejected_count'] 
    
    print("\n--- Integration Summary ---")
    print(f"Total samples in review file: {total_samples}")
    print(f"Approved samples integrated: {approved_count}")
    print(f"Rejected samples (initial analysis): {rejected_count}") 
    print(f"Explicitly approved: {analysis['explicit_approvals_count']}")
    print(f"Explicitly rejected: {analysis['explicit_rejections_count']}")
    print(f"Score-based approvals (>=3 quality & naturalness): {analysis['score_based_approvals_count']}")
    print("---------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Integrate human-approved augmented samples into the dataset.")
    parser.add_argument("--review-file", type=str, required=True,
                        help="Path to the reviewed CSV file (e.g., data/output/review/reviewed_samples.csv)")
    parser.add_argument("--output", type=str, default="data/output/",
                        help="Output directory for final dataset and report")
    parser.add_argument("--format", type=str, default="jsonl", choices=["jsonl", "csv"],
                        help="Output format for the final dataset (jsonl or csv)")
    parser.add_argument("--strategy", type=str, default="balanced",
                        choices=["balanced", "all_approved", "score_only"],
                        help="Integration strategy for selecting samples")

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).mkdir(parents=True, exist_ok=True)

    if not Path(args.review_file).exists():
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

if __name__ == "__main__":
    main()