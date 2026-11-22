"""Summarize all planned paired development episodes, including incomplete ones."""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir', type=Path)
    a = p.parse_args()
    spec = json.loads((a.run_dir / 'config.json').read_text())
    audit_path = a.run_dir / 'semantic_audits.json'
    audits = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    types_path = a.run_dir / 'offline_task_types.json'
    task_types = {(r['domain'], r['task']): r['infeasible']
                  for r in json.loads(types_path.read_text())} if types_path.exists() else {}
    rows = []
    for domain, tasks in spec['tasks'].items():
        for task in tasks:
            for condition in spec['conditions']:
                folder = a.run_dir / condition / domain / task
                result = folder / 'result.json'
                row = dict(domain=domain, task=task, condition=condition, status='not_started')
                if result.exists():
                    row.update(json.loads(result.read_text()))
                selection_path = folder / 'selection.json'
                if selection_path.exists():
                    selection = json.loads(selection_path.read_text())
                    row['selected_skill_count'] = len(selection['selected_skills'])
                    row['mapping_used'] = selection['mapping_used']
                    if 'decision' in selection:
                        decision = selection['decision']
                        if 'operation_constraint_rejections' in decision:
                            row['operation_constraint_rejections'] = decision['operation_constraint_rejections']
                            row['retrieval_requirement_effects'] = [r['effect'] for r in decision['requirements']]
                if row['status'] in ('completed', 'evaluation_error', 'error') and (
                    folder / 'skill_usage_summary.json').exists() and (folder / 'traj.jsonl').exists():
                    usage = json.loads((folder / 'skill_usage_summary.json').read_text())
                    traj = [json.loads(s) for s in (folder / 'traj.jsonl').read_text().splitlines()]
                    row['recorded_actions'] = len(traj)
                    calls = json.loads((folder / 'api/calls.json').read_text())
                    row['recorded_model_calls'] = len(calls)
                    row['recorded_total_tokens'] = sum(
                        (r.get('usage') or {}).get('total_tokens', 0) for r in calls)
                    row['adjacent_repeated_actions'] = sum(
                        x['action'] == y['action'] for x, y in zip(traj, traj[1:]))
                    for field in ('load_skill_calls', 'skill_branch_successes', 'total_loaded_images',
                                  'total_selected_state_views', 'stage1_visual_reference_needed_counts'):
                        row[field] = usage[field]
                    requests = sorted((folder / 'api').glob('request_*.json'))
                    stage_counts = dict(main=0, stage1=0, stage2=0)
                    conditioned = dict(main=0, stage1=0, stage2=0)
                    for request in requests:
                        payload = json.loads(request.read_text())
                        system = payload['messages'][0]['content']
                        stage = 'stage1' if ('You are inside Stage 1' in system or 'CAPABILITY_EVIDENCE_OPTIONS' in system) else (
                            'stage2' if 'You are inside Stage 2' in system else 'main')
                        stage_counts[stage] += 1
                        conditioned[stage] += ('Additional functional reference' in system or
                                               'CAPABILITY_EVIDENCE_OPTIONS' in system)
                    row['request_stages'] = stage_counts
                    row['functional_context_requests'] = conditioned
                    invocations = json.loads((folder / 'skill_invocations.json').read_text())['invocations']
                    counts = Counter((r['step'], r['trigger_skill_name']) for r in invocations)
                    row['same_step_repeat_calls'] = sum(n - 1 for n in counts.values())
                    row['branch_round_errors'] = sum(
                        r.get('status') in ('invalid_state_view_selection', 'state_view_selection_missing_identifiers')
                        for invocation in invocations for r in invocation['rounds'])
                    decisions = folder / 'capability_decisions.jsonl'
                    if decisions.exists():
                        records = [json.loads(s) for s in decisions.read_text().splitlines()]
                        row['structured_decisions'] = len(records)
                        row['call_decisions'] = sum(r['decision'] == 'call' for r in records)
                        row['unconfirmed_effects'] = sum(
                            effect['status'] == 'unconfirmed' for r in records for effect in r['requirements'])
                    evidence = folder / 'capability_evidence_decisions.jsonl'
                    if evidence.exists():
                        records = [json.loads(s) for s in evidence.read_text().splitlines()]
                        row['evidence_decisions'] = len(records)
                        row['no_image_decisions'] = sum(not r['selections'] for r in records)
                        row['evidence_options_selected'] = sum(len(r['selections']) for r in records)
                audit = audits.get(f'{condition}/{domain}/{task}')
                row['audited_score'] = audit['score'] if audit else (
                    row.get('score') if row['status'] == 'completed' else None)
                row['audited_status'] = audit['status'] if audit else row['status']
                row['infeasible'] = task_types.get((domain, task))
                rows.append(row)
    groups = {}
    for condition in spec['conditions']:
        completed = [r for r in rows if r['condition'] == condition and r['audited_score'] is not None]
        groups[condition] = dict(completed=len(completed), planned=sum(map(len, spec['tasks'].values())),
            mean_score_completed_only=sum(r['audited_score'] for r in completed) / len(completed) if completed else None,
            audited_task_failures=sum(r['audited_status'] == 'task_failure' for r in completed))
    report = dict(development_only=True, complete=all(r['audited_status'] in
                  ('completed', 'task_failure', 'environment_error') for r in rows),
                  groups=groups, rows=rows,
                  limits=['Three development tasks do not establish generalization or statistical significance.',
                          'Functional context in a request proves availability, not model use or causal benefit.',
                          'No branch calls means Stage-1/2 are untested online in that episode.'])
    (a.run_dir / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    main()
