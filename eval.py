"""
eval.py — LogMind Evaluation Harness

Runs all 10 labeled sample logs through the agent and scores accuracy.
Measures: category accuracy, flag accuracy, hallucination rate, avg confidence.
"""

import json
import os
from agent import triage_log


def run_eval():
    """Runs all 10 labeled logs through the agent and scores accuracy."""

    with open('data/eval_labels.json') as f:
        labels = json.load(f)

    results = []
    correct_category = 0
    correct_flag = 0
    total_confidence = 0.0
    hallucination_free = 0

    for label in labels:
        log_path = f"data/sample_logs/{label['log_file']}"
        with open(log_path) as f:
            log_text = f.read()

        print(f"Triaging: {label['log_file']}...")
        result = triage_log(log_text)

        # Score category
        actual_category = None
        was_flagged = result.get('triage_outcome') == 'flag_uncertainty'

        if result.get('classification'):
            actual_category = result['classification'].get('category')
            conf = result['classification'].get('confidence', 0)
            total_confidence += conf

        category_correct = actual_category == label['expected_category'] or (label['expected_category'] is None and was_flagged)
        flag_correct = was_flagged == label['should_flag']

        if category_correct:
            correct_category += 1
        if flag_correct:
            correct_flag += 1

        # Hallucination check: did agent cite a real file from the log?
        no_hallucination = True
        if result.get('parsed') and result['parsed'].get('file'):
            cited_file = result['parsed']['file']
            if cited_file and cited_file not in log_text:
                no_hallucination = False
        if no_hallucination:
            hallucination_free += 1

        results.append({
            'log_file': label['log_file'],
            'expected': label['expected_category'],
            'actual': actual_category,
            'category_correct': category_correct,
            'should_flag': label['should_flag'],
            'was_flagged': was_flagged,
            'flag_correct': flag_correct,
            'no_hallucination': no_hallucination
        })

    total = len(labels)
    avg_conf = total_confidence / total if total > 0 else 0.0

    report = f"""===== LogMind Eval Report =====
Total logs:           {total}
Correct category:     {correct_category}/{total}  ({int(correct_category / total * 100)}%)
Flag accuracy:        {correct_flag}/{total}  (flagged correctly)
Avg confidence:       {avg_conf:.2f}
Hallucination-free:   {hallucination_free}/{total}  (cited real files only)

Per-log breakdown:
"""

    for r in results:
        status = 'PASS' if r['category_correct'] and r['flag_correct'] else 'FAIL'
        report += f"  [{status}] {r['log_file']}: expected={r['expected']}, got={r['actual']}\n"

    report += """
Known failure modes:
  [document what actually went wrong after running this]

Recovery paths:
  - confidence < 0.6         -> flag_uncertainty fires, no report generated
  - file + error_type null   -> flag_uncertainty fires
  - log < 5 lines            -> flag_uncertainty fires
  - missing stack trace      -> flag_uncertainty fires
"""

    print(report)
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/eval_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print('Saved to outputs/eval_report.txt')


if __name__ == '__main__':
    run_eval()
