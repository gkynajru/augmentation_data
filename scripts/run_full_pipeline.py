#!/usr/bin/env python3
"""
Quick Start Script - Vietnamese SLU Data Augmentation
====================================================

This script runs the complete augmentation pipeline with example data.
Perfect for testing and getting familiar with the system.

Usage:
    python scripts/run_full_pipeline.py                    # Use example data
    python scripts/run_full_pipeline.py --input your.jsonl # Use your data
    python scripts/run_full_pipeline.py --demo             # Demo mode with auto-approval
"""

import os
import sys
import argparse
import json
import subprocess
import webbrowser
from pathlib import Path
from datetime import datetime

def ensure_directories():
    """Create necessary directories"""
    dirs = [
        "data/input",
        "data/output/augmented", 
        "data/output/review",
        "data/output/final",
        "logs"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        
    print("✅ Created directory structure")

def check_requirements():
    """Check if required packages are installed"""
    try:
        import pandas
        import yaml
        import tqdm
        print("✅ All requirements satisfied")
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def create_demo_data():
    """Create demo data if no input file provided"""
    demo_file = "data/input/demo_data.jsonl"
    
    demo_samples = [
        {
            "sentence": "bật đèn phòng khách",
            "intent": "bật thiết bị",
            "entities": [
                {"type": "device", "filler": "đèn"},
                {"type": "living_space", "filler": "phòng khách"}
            ]
        },
        {
            "sentence": "tắt máy lạnh",
            "intent": "tắt thiết bị", 
            "entities": [{"type": "device", "filler": "máy lạnh"}]
        },
        {
            "sentence": "tăng độ sáng đèn bàn",
            "intent": "tăng độ sáng của thiết bị",
            "entities": [{"type": "device", "filler": "đèn bàn"}]
        },
        {
            "sentence": "giảm âm lượng radio trong phòng ngủ",
            "intent": "giảm âm lượng của thiết bị",
            "entities": [
                {"type": "device", "filler": "radio"},
                {"type": "living_space", "filler": "phòng ngủ"}
            ]
        },
        {
            "sentence": "kiểm tra tình trạng điều hòa",
            "intent": "kiểm tra tình trạng thiết bị",
            "entities": [{"type": "device", "filler": "điều hòa"}]
        }
    ]
    
    with open(demo_file, 'w', encoding='utf-8') as f:
        for sample in demo_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ Created demo data: {demo_file}")
    return demo_file

def run_generation(input_file, test_mode=False):
    """Run augmentation generation"""
    print("\n🔄 Step 1: Generating augmentations...")
    
    cmd = [
        sys.executable, "scripts/1_generate_augmentations.py",
        "--input", input_file,
        "--output", "data/output"
    ]
    
    if test_mode:
        cmd.append("--test-mode")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print("✅ Generation completed successfully!")
            print(result.stdout)
            
            # Find the generated review file
            import glob
            review_files = glob.glob("data/output/review/review_samples_*.csv")
            if review_files:
                return max(review_files)  # Get latest file
            else:
                print("❌ No review file found")
                return None
        else:
            print("❌ Generation failed:")
            print(result.stderr)
            return None
            
    except Exception as e:
        print(f"❌ Error running generation: {e}")
        return None

def create_demo_review(review_file):
    """Create demo review with auto-approval for demonstration"""
    print("\n📝 Creating demo review (auto-approving samples)...")
    
    try:
        import pandas as pd
        
        # Load the review file
        df = pd.read_csv(review_file)
        
        # Auto-approve samples with demo scores
        for i, row in df.iterrows():
            # Give decent scores to most samples
            df.at[i, 'Quality_Score_1_5'] = 4
            df.at[i, 'Naturalness_Score_1_5'] = 4
            df.at[i, 'Approved_Yes_No'] = 'yes'
            df.at[i, 'Comments'] = 'Auto-approved for demo'
        
        # Save the reviewed file
        reviewed_file = review_file.replace('review_samples_', 'reviewed_samples_')
        df.to_csv(reviewed_file, index=False, encoding='utf-8')
        
        print(f"✅ Demo review created: {reviewed_file}")
        return reviewed_file
        
    except Exception as e:
        print(f"❌ Error creating demo review: {e}")
        return None

def open_dashboard(review_file):
    """Open the review dashboard"""
    print(f"\n🌐 Opening review dashboard...")
    
    dashboard_path = os.path.abspath("dashboard/index.html")
    
    if os.path.exists(dashboard_path):
        try:
            webbrowser.open(f"file://{dashboard_path}")
            print(f"✅ Dashboard opened in browser")
            print(f"📝 Load this file in the dashboard: {review_file}")
            return True
        except Exception as e:
            print(f"❌ Error opening dashboard: {e}")
            print(f"Please manually open: {dashboard_path}")
            return False
    else:
        print(f"❌ Dashboard not found: {dashboard_path}")
        return False

def run_integration(reviewed_file):
    """Run data integration"""
    print("\n🔗 Step 3: Integrating approved data...")
    
    cmd = [
        sys.executable, "scripts/3_integrate_approved_data.py",
        "--review-file", reviewed_file,
        "--output", "data/output/final"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print("✅ Integration completed successfully!")
            print(result.stdout)
            
            # Find the final dataset file
            import glob
            final_files = glob.glob("data/output/final/final_dataset_*.jsonl")
            if final_files:
                return max(final_files)
            else:
                return None
        else:
            print("❌ Integration failed:")
            print(result.stderr)
            return None
            
    except Exception as e:
        print(f"❌ Error running integration: {e}")
        return None

def print_summary(input_file, final_file):
    """Print pipeline summary"""
    print("\n" + "="*60)
    print("🎉 VIETNAMESE SLU AUGMENTATION PIPELINE COMPLETE!")
    print("="*60)
    
    print(f"\n📁 Files Created:")
    print(f"   Input: {input_file}")
    print(f"   Final Dataset: {final_file}")
    
    if final_file and os.path.exists(final_file):
        # Count lines in final file
        with open(final_file, 'r', encoding='utf-8') as f:
            final_count = sum(1 for line in f if line.strip())
        
        # Count lines in input file  
        with open(input_file, 'r', encoding='utf-8') as f:
            input_count = sum(1 for line in f if line.strip())
        
        augmentation_ratio = (final_count / input_count - 1) * 100 if input_count > 0 else 0
        
        print(f"\n📊 Results:")
        print(f"   Original samples: {input_count}")
        print(f"   Final dataset size: {final_count}")
        print(f"   Augmentation ratio: +{augmentation_ratio:.1f}%")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Upload {final_file} to your training environment")
    print(f"   2. Load into your training pipeline")
    print(f"   3. Train your model and monitor improvements!")
    print(f"   4. Expected: F1 Macro 21% → 45-60% improvement")

def main():
    parser = argparse.ArgumentParser(description='Vietnamese SLU Augmentation Quick Start')
    
    parser.add_argument('--input', '-i', 
                       help='Input JSONL file (uses demo data if not provided)')
    parser.add_argument('--demo', action='store_true',
                       help='Demo mode: auto-approve samples for testing')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode: process only first 50 samples')
    parser.add_argument('--skip-dashboard', action='store_true',
                       help='Skip opening the dashboard')
    
    args = parser.parse_args()
    
    print("🇻🇳 Vietnamese SLU Data Augmentation - Quick Start")
    print("="*55)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Ensure directories exist
    ensure_directories()
    
    # Determine input file
    if args.input:
        if not os.path.exists(args.input):
            print(f"❌ Input file not found: {args.input}")
            sys.exit(1)
        input_file = args.input
        print(f"📁 Using input file: {input_file}")
    else:
        input_file = create_demo_data()
        print("📁 Using demo data")
    
    # Step 1: Generate augmentations
    review_file = run_generation(input_file, args.test_mode)
    if not review_file:
        print("❌ Pipeline failed at generation step")
        sys.exit(1)
    
    # Step 2: Handle review process
    if args.demo:
        # Auto-approve for demo
        reviewed_file = create_demo_review(review_file)
        if not reviewed_file:
            print("❌ Pipeline failed at demo review step")
            sys.exit(1)
    else:
        # Open dashboard for manual review
        if not args.skip_dashboard:
            dashboard_opened = open_dashboard(review_file)
        
        print(f"\n⏸️  MANUAL REVIEW REQUIRED")
        print(f"   1. Review samples in the dashboard or CSV file")
        print(f"   2. Fill in Quality_Score_1_5 and Naturalness_Score_1_5")
        print(f"   3. Set Approved_Yes_No to 'yes' or 'no'")
        print(f"   4. Save the file as 'reviewed_samples_*.csv'")
        print(f"   5. Run integration manually:")
        print(f"      python scripts/3_integrate_approved_data.py --review-file <reviewed_file>")
        print(f"\n   Review file: {review_file}")
        return
    
    # Step 3: Integrate approved data
    final_file = run_integration(reviewed_file)
    if not final_file:
        print("❌ Pipeline failed at integration step")
        sys.exit(1)
    
    # Summary
    print_summary(input_file, final_file)

if __name__ == "__main__":
    main()