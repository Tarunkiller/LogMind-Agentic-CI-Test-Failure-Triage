"""
agent.py — LogMind Core Agent

Calls the Anthropic API with tool_use and manages the 4-step
triage pipeline: parse_log → classify_failure → generate_bug_report or flag_uncertainty.

Falls back to local-only mode if ANTHROPIC_API_KEY has no credits.
"""

import json
import os
from dotenv import load_dotenv
from tools import parse_log, classify_failure, generate_bug_report, flag_uncertainty

load_dotenv()

_USE_LOCAL = False  # set to True to skip API calls entirely

try:
    from anthropic import Anthropic, BadRequestError as _AnthropicBadRequest
    client = Anthropic()
    MODEL = 'claude-sonnet-4-20250514'
except Exception:
    _USE_LOCAL = True
    _AnthropicBadRequest = Exception

# ---------------------------------------------------------------------------
# Tool definitions — passed to Claude so it knows what tools exist
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "parse_log",
        "description": "Extracts structured fields from raw CI log text. Returns error_type, file, line, stack_trace, test_name. Returns null for unknown fields — never guesses.",
        "input_schema": {"type": "object", "properties": {"log_text": {"type": "string", "description": "Raw CI log content"}}, "required": ["log_text"]}
    },
    {
        "name": "classify_failure",
        "description": "Classifies failure into a category with a confidence score (0.0-1.0). Categories: logic_bug, environment_issue, flaky_test, dependency_error, timeout, unknown.",
        "input_schema": {"type": "object", "properties": {"parsed_data": {"type": "object", "description": "Output from parse_log"}}, "required": ["parsed_data"]}
    },
    {
        "name": "generate_bug_report",
        "description": "Generates and saves a structured bug report as JSON. Only call when confidence >= 0.6 and key fields are present.",
        "input_schema": {"type": "object", "properties": {"classification": {"type": "object"}, "parsed_data": {"type": "object"}, "original_log": {"type": "string"}}, "required": ["classification", "parsed_data", "original_log"]}
    },
    {
        "name": "flag_uncertainty",
        "description": "Flag this log for manual review when: confidence < 0.6, both file and error_type are null, log has fewer than 5 lines, or stack trace is missing. Do NOT generate a bug report when uncertain.",
        "input_schema": {"type": "object", "properties": {"reason": {"type": "string", "description": "Explain why the log cannot be triaged automatically"}}, "required": ["reason"]}
    }
]

TOOL_MAP = {
    "parse_log": lambda args: parse_log(args["log_text"]),
    "classify_failure": lambda args: classify_failure(args["parsed_data"]),
    "generate_bug_report": lambda args: generate_bug_report(args["classification"], args["parsed_data"], args["original_log"]),
    "flag_uncertainty": lambda args: flag_uncertainty(args["reason"])
}

SYSTEM_PROMPT = """You are LogMind, an expert CI test failure triage agent.
Your job is to analyze CI logs and produce structured bug reports.

You MUST follow this exact pipeline for every log:
1. Call parse_log to extract structured fields
2. Call classify_failure with the parsed result
3. Check the confidence score:
   - If confidence < 0.6, OR both file and error_type are null,
     OR the log has fewer than 5 lines, OR stack trace is missing:
     → call flag_uncertainty and STOP
   - Otherwise: call generate_bug_report

Never skip a step. Never hallucinate file paths or error types not present in the log.
Admitting uncertainty is correct behavior — it is better than a wrong bug report."""


def _triage_local(log_text: str) -> dict:
    """
    Runs the triage pipeline locally without any API calls.
    Used as fallback when Anthropic API has no credits.
    Follows the same logic the agent would enforce via tool_use.
    """
    results = {'parsed': None, 'classification': None, 'final': None, 'mode': 'local'}

    # Step 1: parse
    parsed = parse_log(log_text)
    results['parsed'] = parsed

    # Step 2: classify
    classification = classify_failure(parsed)
    results['classification'] = classification

    lines = log_text.strip().split('\n')
    confidence = classification.get('confidence', 0)
    file_val = parsed.get('file')
    error_val = parsed.get('error_type')
    stack_val = parsed.get('stack_trace')

    # Step 3: uncertainty check (mirrors SYSTEM_PROMPT rules)
    should_flag = (
        confidence < 0.6
        or (file_val is None and error_val is None)
        or len(lines) < 5
        or stack_val is None
    )

    if should_flag:
        reasons = []
        if confidence < 0.6:
            reasons.append(f"confidence {confidence:.2f} < 0.6")
        if file_val is None and error_val is None:
            reasons.append("both file and error_type are null")
        if len(lines) < 5:
            reasons.append(f"log has only {len(lines)} lines (< 5)")
        if stack_val is None:
            reasons.append("no stack trace found")
        reason_str = "; ".join(reasons)
        results['final'] = flag_uncertainty(reason_str)
        results['triage_outcome'] = 'flag_uncertainty'
    else:
        results['final'] = generate_bug_report(classification, parsed, log_text)
        results['triage_outcome'] = 'generate_bug_report'

    return results


def _triage_api(log_text: str) -> dict:
    """
    Runs the full 4-step triage pipeline via Anthropic tool_use API.
    Returns a dict with all intermediate outputs and final result.
    """
    messages = [{'role': 'user', 'content': f'Triage this CI log:\n\n{log_text}'}]
    results = {'parsed': None, 'classification': None, 'final': None, 'mode': 'api'}

    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=2000, system=SYSTEM_PROMPT,
            tools=TOOLS, messages=messages
        )
        messages.append({'role': 'assistant', 'content': response.content})

        if response.stop_reason == 'end_turn':
            break

        tool_results = []
        for block in response.content:
            if block.type != 'tool_use':
                continue
            result = TOOL_MAP[block.name](block.input) if block.name in TOOL_MAP else {'error': f'Unknown tool: {block.name}'}

            if block.name == 'parse_log':               results['parsed'] = result
            elif block.name == 'classify_failure':       results['classification'] = result
            elif block.name in ('generate_bug_report', 'flag_uncertainty'):
                results['final'] = result
                results['triage_outcome'] = block.name

            tool_results.append({'type': 'tool_result', 'tool_use_id': block.id, 'content': json.dumps(result)})

        if tool_results:
            messages.append({'role': 'user', 'content': tool_results})
        else:
            break

    return results


def triage_log(log_text: str) -> dict:
    """
    Runs the full triage pipeline on a single CI log.
    Uses Anthropic API if available, otherwise falls back to local mode.
    Returns a dict with all intermediate outputs and final result.
    """
    if _USE_LOCAL:
        return _triage_local(log_text)
    try:
        return _triage_api(log_text)
    except Exception as e:
        err = str(e)
        if 'credit' in err.lower() or '400' in err or '401' in err or '403' in err:
            print(f"  [API unavailable, using local mode: {err[:60]}...]")
            return _triage_local(log_text)
        raise
