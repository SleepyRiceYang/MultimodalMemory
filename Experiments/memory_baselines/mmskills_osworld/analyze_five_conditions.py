"""Merge the four reclassified arms with the reconciled W/O Images 40-task run."""
from __future__ import annotations

import csv
import itertools
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / 'Experiments/memory_baselines/artifacts/mmskills_osworld'
OLD = ART / 'analysis_20261111_v2/statistics.json'
WOI = ART / 'formal_40_without_images_newhost_v1'
OUT = ART / 'analysis_20261111_v3'
CONDITIONS = ('no_skill', 'text_skill', 'visual_skill', 'visual_skill_retrieval',
              'visual_skill_without_images')


def exact_mcnemar(gains: int, losses: int) -> float:
    n = gains + losses
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(gains, losses) + 1)) / 2**n) if n else 1.


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, running = [0.] * len(p_values), 0.
    for rank, i in enumerate(order):
        running = min(1., max(running, p_values[i] * (len(p_values) - rank)))
        adjusted[i] = running
    return adjusted


def main() -> None:
    old = json.loads(OLD.read_text())
    new = json.loads((WOI / 'reconciled_results.json').read_text())
    run_config = json.loads((WOI / 'config.json').read_text())
    expected = {(domain, task) for domain, ids in run_config['tasks'].items() for task in ids}
    if run_config['conditions'] != ['visual_skill_without_images'] or len(expected) != 40:
        raise ValueError('W/O Images config is not the frozen single-condition 40-task run')
    new_keys = {(row['domain'], row['task']) for row in new}
    if len(new) != 40 or new_keys != expected:
        raise ValueError('W/O Images reconciled rows do not cover the frozen split exactly')

    rows = list(old['rows'])
    for row in new:
        folder = WOI / row['source']
        initial = WOI / 'visual_skill_without_images' / row['domain'] / row['task'] / 'result.json'
        first_result = json.loads(initial.read_text())
        rows.append({
            'condition': 'visual_skill_without_images',
            'domain': row['domain'], 'task': row['task'],
            'original_status': row['initial_status'],
            'original_score': first_result.get('score'),
            'raw_score': row['raw_score'], 'outcome': row['outcome'], 'score': row['score'],
            'reason': row['score_basis'], 'evidence': row['evidence'],
            'api_error_count': row['api_errors'], 'artifact_dir': str(folder),
            'source_result_sha256': None, 'steps': row['steps'], 'actions': row['actions'],
            'model_calls': row['model_calls'], 'selected_skills': row['selected_skills'],
            'source': row['source'], 'final_status': row['final_status'],
            'evaluation_status': row['evaluation_status'], 'initial_error': row['initial_error'],
            'initial_api_errors': row['initial_api_errors'],
        })

    records = {condition: {(r['domain'], r['task']): r for r in rows if r['condition'] == condition}
               for condition in CONDITIONS}
    if any(len(group) != 40 for group in records.values()):
        raise ValueError('Each condition must have exactly 40 unique tasks')
    common = {tuple(task) for task in old['common_tasks']}
    scorable = {condition: {key for key, row in group.items() if row['score'] is not None}
                for condition, group in records.items()}
    if set.intersection(*(scorable[c] for c in CONDITIONS)) != common or len(common) != 35:
        raise ValueError('Five-condition common scorable set differs from the original 35-task set')

    summary, domains = {}, {}
    for condition, group in records.items():
        scores = [row['score'] for row in group.values() if row['score'] is not None]
        common_scores = [group[key]['score'] for key in common]
        summary[condition] = {
            'planned': 40, 'scorable': len(scores), 'environment_errors': 40 - len(scores),
            'outcomes': dict(Counter(row['outcome'] for row in group.values())),
            'mean_scorable': sum(scores) / len(scores) if scores else None,
            'sum_scores': sum(scores), 'fixed40_lower_bound': sum(scores) / 40,
            'fixed40_upper_bound': (sum(scores) + 40 - len(scores)) / 40,
            'common_n': len(common), 'common_mean': sum(common_scores) / len(common),
            'common_full_successes': sum(score == 1 for score in common_scores),
        }
        domains[condition] = {}
        for domain in sorted({key[0] for key in group}):
            domain_rows = [row for (d, _), row in group.items() if d == domain]
            domain_scores = [row['score'] for row in domain_rows if row['score'] is not None]
            domains[condition][domain] = {
                'planned': len(domain_rows), 'scorable': len(domain_scores),
                'mean': sum(domain_scores) / len(domain_scores) if domain_scores else None,
                'score_sum': sum(domain_scores),
                'outcomes': dict(Counter(row['outcome'] for row in domain_rows)),
            }

    paired = []
    for a, b in itertools.combinations(CONDITIONS, 2):
        changes = [(records[b][key]['score'], records[a][key]['score']) for key in common]
        gains = sum(x == 1 and y != 1 for x, y in changes)
        losses = sum(x != 1 and y == 1 for x, y in changes)
        paired.append({
            'a': a, 'b': b, 'n': len(common), 'b_gains': gains, 'b_losses': losses,
            'b_minus_a_mean': sum(x - y for x, y in changes) / len(common),
            'p_exact_mcnemar': exact_mcnemar(gains, losses),
        })
    adjusted = holm([pair['p_exact_mcnemar'] for pair in paired])
    for pair, value in zip(paired, adjusted):
        pair['p_holm_10'] = value

    output = {
        'policy': ('Only environment-caused inability to assess is exceptional; unmet goal is score 0. '
                   'Retry artifacts replace setup/API/getter failures when available. W/O Images 404s '
                   'were checked against task output paths and readable expected evaluator inputs.'),
        'verification_status': ('ANALYZED; W/O Images formal 40-task run completed, 4 failed attempts '
                                'retried, all 40 tasks reconciled and scored.'),
        'scorable': sum(item['scorable'] for item in summary.values()),
        'environment_errors': sum(item['environment_errors'] for item in summary.values()),
        'reclassified_goal_failures': old['reclassified_goal_failures'] + 4,
        'summary': summary, 'common_n': len(common),
        'common_tasks': [list(key) for key in sorted(common)], 'paired_tests': paired,
        'domains': domains, 'conditions': list(CONDITIONS), 'rows': rows,
        'sources': {
            'previous_four_group_analysis': str(OLD.relative_to(ART)),
            'without_images_run': str((WOI / 'config.json').relative_to(ART)),
            'without_images_reconciled': str((WOI / 'reconciled_results.json').relative_to(ART)),
            'without_images_summary': str((WOI / 'reconciled_summary.json').relative_to(ART)),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'statistics.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / 'per_task.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False)
                             if isinstance(row.get(key), (list, dict)) else row.get(key)
                             for key in fields})
    if len(rows) != 200 or output['scorable'] != 195 or len(paired) != 10:
        raise ValueError('Five-condition output failed coverage checks')
    print(json.dumps({'rows': len(rows), 'scorable': output['scorable'],
                      'environment_errors': output['environment_errors'],
                      'without_images_mean': summary['visual_skill_without_images']['mean_scorable'],
                      'common_n': len(common), 'paired_tests': len(paired)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
