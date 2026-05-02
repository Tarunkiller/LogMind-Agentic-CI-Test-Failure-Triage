"""
main.py — LogMind CLI Entry Point

Usage:
    python main.py --log data/sample_logs/assertion_error.txt
    python main.py --eval
    python main.py --dir data/sample_logs/
"""

import argparse
import os
import json
from agent import triage_log
from eval import run_eval


def main():
    """CLI entry point for LogMind — CI log triage agent."""
    parser = argparse.ArgumentParser(
        description='LogMind — CI Log Triage Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --log data/sample_logs/assertion_error.txt
  python main.py --eval
  python main.py --dir data/sample_logs/
        """
    )
    parser.add_argument('--log',  help='Path to a single log file to triage')
    parser.add_argument('--eval', action='store_true', help='Run full eval harness on all 10 labeled logs')
    parser.add_argument('--dir',  help='Triage all .txt logs in a directory')
    args = parser.parse_args()

    if args.eval:
        run_eval()

    elif args.log:
        if not os.path.exists(args.log):
            print(f"Error: File not found: {args.log}")
            return
        with open(args.log) as f:
            log_text = f.read()
        print(f"\nTriaging: {args.log}\n{'='*50}")
        result = triage_log(log_text)
        print(json.dumps(result, indent=2))

    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"Error: Directory not found: {args.dir}")
            return
        logs = [f for f in os.listdir(args.dir) if f.endswith('.txt')]
        if not logs:
            print(f"No .txt log files found in {args.dir}")
            return
        print(f"\nTriaging {len(logs)} log(s) in {args.dir}\n{'='*50}")
        for log_file in sorted(logs):
            path = os.path.join(args.dir, log_file)
            print(f'\nTriaging {log_file}...')
            with open(path) as f:
                log_text = f.read()
            result = triage_log(log_text)
            outcome = result.get('triage_outcome', 'unknown')
            print(f'  Outcome: {outcome}')
            if result.get('classification'):
                conf = result['classification'].get('confidence', 0)
                cat = result['classification'].get('category', 'unknown')
                print(f'  Category: {cat}  |  Confidence: {conf:.2f}')
            if outcome == 'flag_uncertainty' and result.get('final'):
                print(f"  Reason: {result['final'].get('reason', '')}")
            elif outcome == 'generate_bug_report' and result.get('final'):
                print(f"  Saved to: {result['final'].get('saved_to', '')}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
