#!/usr/bin/env python
"""
start.py — Launch AI Vigilance with Dynamic Auto-Scaling

This is the recommended way to run AI Vigilance in production or development.
It automatically scales workers based on CPU load and number of active cameras.

Usage:
    python start.py                              # Default: 2-8 workers, auto-scale at 70% CPU
    python start.py --max-workers 16             # Scale up to 16 workers
    python start.py --cpu-threshold 60           # Aggressive scaling (scale at 60% CPU)
    python start.py --disable-autoscale          # Run without auto-scaling

Environment Variables (optional):
    AUTOSCALE_MIN_WORKERS=2
    AUTOSCALE_MAX_WORKERS=8
    AUTOSCALE_CPU_THRESHOLD=70
    AUTOSCALE_MEMORY_THRESHOLD=80
    AUTOSCALE_CAMERA_WORKER_RATIO=0.5

Examples:

1. Development (minimal resources):
   python start.py --max-workers 4 --cpu-threshold 80

2. Production (aggressive scaling):
   export AUTOSCALE_MAX_WORKERS=16
   python start.py --cpu-threshold 60

3. Without auto-scaling (fixed workers):
   python start.py --disable-autoscale
"""

import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Launch AI Vigilance with Dynamic Auto-Scaling",
        epilog="For more options, run: python autoscale.py --help"
    )
    parser.add_argument("--disable-autoscale", action="store_true",
                        help="Disable auto-scaling, use fixed worker count")
    parser.add_argument("--min-workers", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--cpu-threshold", type=int, default=None)
    parser.add_argument("--camera-worker-ratio", type=float, default=None)
    
    args = parser.parse_args()
    
    if args.disable_autoscale:
        # Run directly without autoscaling
        print("\n🚀 Starting AI Vigilance (without auto-scaling)...\n")
        os.system("python app.py")
    else:
        # Run with autoscaling
        cmd = ["python", "autoscale.py"]
        
        if args.min_workers:
            cmd.extend(["--min-workers", str(args.min_workers)])
        if args.max_workers:
            cmd.extend(["--max-workers", str(args.max_workers)])
        if args.cpu_threshold:
            cmd.extend(["--cpu-threshold", str(args.cpu_threshold)])
        if args.camera_worker_ratio:
            cmd.extend(["--camera-worker-ratio", str(args.camera_worker_ratio)])
        
        print("\n🚀 Starting AI Vigilance with Dynamic Auto-Scaling...\n")
        os.system(" ".join(cmd))


if __name__ == "__main__":
    main()
