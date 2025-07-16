#!/usr/bin/env python3
"""
Create Review Dashboard Script
==============================

This script prepares the review dashboard and starts a web server for reviewing
augmented samples.

Usage:
    python scripts/2_create_review_dashboard.py
    python scripts/2_create_review_dashboard.py --port 8080
    python scripts/2_create_review_dashboard.py --review-file data/output/review/review_samples.csv
"""

import os
import sys
import argparse
import webbrowser
import http.server
import socketserver
import threading
import time
from pathlib import Path
import json
import glob

def find_latest_review_file(output_dir: str = "data/output/review") -> str:
    """
    Find the most recent review CSV file
    """
    pattern = os.path.join(output_dir, "review_samples_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Sort by modification time and get the latest
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def check_dashboard_exists() -> bool:
    """
    Check if dashboard HTML file exists
    """
    dashboard_path = "dashboard/index.html"
    return os.path.exists(dashboard_path)

def create_simple_server(port: int = 8000):
    """
    Create a simple HTTP server
    """
    Handler = http.server.SimpleHTTPRequestHandler
    
    class QuietHandler(Handler):
        def log_message(self, format, *args):
            # Only log errors, not every request
            if args[1] != '200':
                super().log_message(format, *args)
    
    with socketserver.TCPServer(("", port), QuietHandler) as httpd:
        print(f"✅ Server running at http://localhost:{port}")
        print(f"   Dashboard: http://localhost:{port}/dashboard/index.html")
        print("\n   Press Ctrl+C to stop the server")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n⏹️  Server stopped")
            httpd.shutdown()

def open_browser_delayed(url: str, delay: float = 1.5):
    """
    Open browser after a delay
    """
    time.sleep(delay)
    webbrowser.open(url)

def print_instructions():
    """
    Print usage instructions
    """
    print("\n" + "="*60)
    print("REVIEW DASHBOARD INSTRUCTIONS")
    print("="*60)
    
    print("\n📋 How to use the dashboard:")
    print("1. The dashboard should open automatically in your browser")
    print("2. If not, go to: http://localhost:8000/dashboard/index.html")
    print("3. Click 'Load CSV' and select your review file")
    print("4. Review each sample:")
    print("   - Rate Quality (1-5): How well the meaning is preserved")
    print("   - Rate Naturalness (1-5): How natural it sounds in Vietnamese")
    print("   - Click Approve ✓ or Reject ✗")
    print("5. Use filters to focus on specific methods or statuses")
    print("6. Click 'Export Results' to save your reviewed data")
    
    print("\n⌨️  Keyboard shortcuts:")
    print("   1-5: Set quality score")
    print("   Q/W/E/R/T: Set naturalness score (1-5)")
    print("   A: Approve sample")
    print("   X: Reject sample")
    print("   N: Next sample")
    print("   P: Previous sample")
    
    print("\n💡 Tips:")
    print("   - Use 'Auto-Approve High Quality' for samples with both scores ≥ 4")
    print("   - The dashboard auto-saves every 30 seconds")
    print("   - Export creates both CSV and JSON files")

def check_prerequisites():
    """
    Check if all required files exist
    """
    issues = []
    
    # Check dashboard
    if not check_dashboard_exists():
        issues.append("❌ Dashboard not found at dashboard/index.html")
    
    # Check for review files
    review_dir = "data/output/review"
    if not os.path.exists(review_dir):
        issues.append(f"❌ Review directory not found: {review_dir}")
    else:
        latest_review = find_latest_review_file()
        if not latest_review:
            issues.append("❌ No review CSV files found")
            issues.append("   Run: python scripts/1_generate_augmentations.py first")
    
    if issues:
        print("\n⚠️  Prerequisites check failed:")
        for issue in issues:
            print(f"   {issue}")
        return False
    
    return True

def create_dashboard_config(review_file: str = None):
    """
    Create a configuration file for the dashboard
    """
    config = {
        'defaultReviewFile': review_file,
        'autoLoadFile': review_file is not None,
        'port': 8000,
        'features': {
            'autoSave': True,
            'autoSaveInterval': 30,
            'showProgressBar': True,
            'enableKeyboardShortcuts': True
        }
    }
    
    config_path = "dashboard/config.json"
    os.makedirs("dashboard", exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created dashboard configuration: {config_path}")

def main():
    parser = argparse.ArgumentParser(description='Launch the review dashboard')
    
    parser.add_argument('--port', '-p', type=int, default=8000,
                       help='Port for the web server (default: 8000)')
    parser.add_argument('--review-file', '-f', type=str,
                       help='Path to review CSV file (auto-detects if not specified)')
    parser.add_argument('--no-browser', action='store_true',
                       help='Don\'t open browser automatically')
    parser.add_argument('--config-only', action='store_true',
                       help='Only create config file, don\'t start server')
    
    args = parser.parse_args()
    
    print("Vietnamese SLU Review Dashboard")
    print("="*50)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Please fix the issues above before running the dashboard")
        sys.exit(1)
    
    # Find review file
    review_file = args.review_file
    if not review_file:
        review_file = find_latest_review_file()
        if review_file:
            print(f"✅ Found latest review file: {review_file}")
        else:
            print("⚠️  No review file specified or found")
            print("   You'll need to load one manually in the dashboard")
    
    # Create configuration
    create_dashboard_config(review_file)
    
    if args.config_only:
        print("\n✅ Configuration created. Run without --config-only to start server")
        return
    
    # Print instructions
    print_instructions()
    
    # Start browser in background (if requested)
    if not args.no_browser:
        url = f"http://localhost:{args.port}/dashboard/index.html"
        browser_thread = threading.Thread(target=open_browser_delayed, args=(url,))
        browser_thread.daemon = True
        browser_thread.start()
    
    # Start server
    print(f"\n🚀 Starting web server on port {args.port}...")
    
    try:
        create_simple_server(args.port)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n❌ Port {args.port} is already in use!")
            print(f"   Try a different port: python {sys.argv[0]} --port 8080")
        else:
            print(f"\n❌ Server error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()