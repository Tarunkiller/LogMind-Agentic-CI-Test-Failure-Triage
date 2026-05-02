"""
tools.py — LogMind Tool Implementations

4 pure Python functions used by the agent pipeline.
No Anthropic API calls happen here — these are data transformation
functions that Claude will invoke via tool_use.
"""

import json
import os
from datetime import datetime


# ---------------------------------------------------------------------------
# Tool 1: parse_log
# ---------------------------------------------------------------------------

def parse_log(log_text: str) -> dict:
    """
    Extracts structured fields from raw CI log text.
    Returns null for any field that cannot be determined.
    Never guesses — missing data is returned as None.

    Returns:
        dict with keys: error_type, file, line, stack_trace, test_name
    """
    result = {
        "error_type": None,
        "file": None,
        "line": None,
        "stack_trace": None,
        "test_name": None,
        "_raw_log": log_text
    }

    lines = log_text.strip().split('\n')

    for line in lines:
        # Extract test name from pytest output
        if 'FAILED' in line and '::' in line:
            result['test_name'] = line.split('::')[-1].split(' ')[0]

        # Better file and line extraction for pytest or traceback
        if 'tests/' in line and ':' in line:
            parts = line.strip().split(':')
            for i, part in enumerate(parts):
                if part.strip().endswith('.py'):
                    result['file'] = part.strip().split()[-1]
                    if i + 1 < len(parts):
                        try:
                            result['line'] = int(parts[i+1])
                        except ValueError:
                            pass

        if line.strip().startswith('File ') and ', line ' in line:
            parts = line.strip().split(',')
            result['file'] = parts[0].replace('File ', '').strip().strip('"').strip("'")
            try:
                result['line'] = int(parts[1].replace(' line ', '').strip())
            except ValueError:
                pass

        # Extract error type (last line of traceback or error message)
        if any(e in line for e in ['Error', 'Exception', 'Warning', 'Timeout']):
            if ': ' in line:
                result['error_type'] = line.split(':')[0].strip().split()[-1]
            else:
                for word in line.split():
                    if any(e in word for e in ['Error', 'Exception', 'Warning', 'Timeout']):
                        result['error_type'] = word.strip()

    # Extract stack trace block
    if 'Traceback' in log_text:
        start = log_text.find('Traceback')
        result['stack_trace'] = log_text[start:start + 1500]
    else:
        result['stack_trace'] = log_text[:2000]

    return result


# ---------------------------------------------------------------------------
# Tool 2: classify_failure
# ---------------------------------------------------------------------------

CATEGORIES = [
    "logic_bug",         # wrong output, assertion failed
    "environment_issue", # missing package, CUDA, config
    "flaky_test",        # intermittent, network, timing
    "dependency_error",  # version conflict, import fail
    "timeout",           # test exceeded time limit
    "unknown"            # insufficient information
]


def classify_failure(parsed_data: dict) -> dict:
    """
    Classifies failure into a category with confidence score.
    Confidence < 0.6 should trigger flag_uncertainty downstream.

    Returns:
        dict with keys: category, confidence, reasoning
    """
    error = (parsed_data.get('error_type') or '').lower()
    trace = (parsed_data.get('stack_trace') or '').lower()
    raw = (parsed_data.get('_raw_log') or '').lower()
    combined = error + ' ' + trace + ' ' + raw

    # Order of checks is highly important! Check more specific/environment issues first.
    if any(x in combined for x in ['modulenotfounderror', 'importerror', 'cuda', 'no module', 'fixture']):
        return {
            "category": "environment_issue",
            "confidence": 0.88,
            "reasoning": "Import, fixture, or environment setup failure — missing dependency or misconfigured runtime."
        }

    if any(x in combined for x in ['timeout', 'timed out', 'exceeded']):
        return {
            "category": "timeout",
            "confidence": 0.90,
            "reasoning": "Test exceeded time limit — likely slow external call or infinite loop."
        }

    if any(x in combined for x in ['connection refused', 'connectionrefusederror', 'intermittent', 'flaky', 'network']):
        return {
            "category": "flaky_test",
            "confidence": 0.75,
            "reasoning": "Network or timing-dependent failure — likely flaky test or unstable dependency."
        }

    if any(x in combined for x in ['versionconflict', 'requires', 'incompatible']):
        return {
            "category": "dependency_error",
            "confidence": 0.82,
            "reasoning": "Package version conflict detected — dependency resolution needed."
        }

    if 'nonetype' in combined or 'attributeerror' in combined:
        return {
            "category": "logic_bug",
            "confidence": 0.78,
            "reasoning": "NoneType or AttributeError — null reference or unexpected object state."
        }

    if 'assertionerror' in combined or 'assert ' in combined:
        return {
            "category": "logic_bug",
            "confidence": 0.85,
            "reasoning": "AssertionError indicates test expectation mismatch — likely logic bug."
        }

    # Fallback — not enough signal
    return {
        "category": "unknown",
        "confidence": 0.35,
        "reasoning": "Insufficient signal to classify failure type confidently."
    }


# ---------------------------------------------------------------------------
# Tool 3: generate_bug_report
# ---------------------------------------------------------------------------

def generate_bug_report(classification: dict, parsed_data: dict, original_log: str) -> dict:
    """
    Generates structured bug report and saves it to disk.
    Only called when confidence >= 0.6 and key fields are present.

    Returns:
        dict with full bug report fields
    """
    severity_map = {
        "logic_bug": "high",
        "environment_issue": "medium",
        "flaky_test": "low",
        "dependency_error": "medium",
        "timeout": "medium",
        "unknown": "low"
    }

    test_name = parsed_data.get('test_name') or 'unknown_test'
    category = classification.get('category', 'unknown')

    report = {
        "title": f"[{category.upper()}] Failure in {test_name}",
        "severity": severity_map.get(category, "medium"),
        "category": category,
        "affected_file": parsed_data.get("file"),
        "affected_line": parsed_data.get("line"),
        "root_cause": classification.get("reasoning"),
        "error_type": parsed_data.get("error_type"),
        "steps_to_reproduce": "Run the failing test in isolation. Check stack trace for exact failure point.",
        "suggested_fix": _suggest_fix(category),
        "confidence": classification.get("confidence"),
        "generated_at": datetime.now().isoformat()
    }

    # Save to disk
    os.makedirs('outputs/reports', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'outputs/reports/{test_name}_{ts}.json'
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)

    report['saved_to'] = path
    return report


def _suggest_fix(category: str) -> str:
    """Returns a suggested fix string based on the failure category."""
    fixes = {
        "logic_bug": "Review assertion logic and expected values. Add debug prints around failure point.",
        "environment_issue": "Check installed packages match requirements.txt. Verify CUDA/GPU availability.",
        "flaky_test": "Add retry logic or mock the external dependency. Consider marking as xfail.",
        "dependency_error": "Run pip install -r requirements.txt in a fresh virtualenv.",
        "timeout": "Profile the test for slow operations. Consider increasing timeout or mocking slow calls.",
        "unknown": "Manual investigation required — insufficient log data for automated suggestion."
    }
    return fixes.get(category, 'Manual review required.')


# ---------------------------------------------------------------------------
# Tool 4: flag_uncertainty
# ---------------------------------------------------------------------------

def flag_uncertainty(reason: str) -> dict:
    """
    Called instead of generate_bug_report when the agent
    cannot confidently determine the failure root cause.
    This is correct behavior, not a failure state.

    Returns:
        dict signaling manual review is needed
    """
    return {
        "flagged": True,
        "reason": reason,
        "recommendation": "manual_review",
        "message": "Agent confidence too low to generate reliable bug report. Human review required."
    }
