"""Independent iteration: preserve task effects across local capability decisions."""
import json
from pathlib import Path

from capability_v3_agent import CapabilityV3Agent


class CapabilityV4Agent(CapabilityV3Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._goal_contract = None

    def _build_openai_messages(self, system_text, contents):
        messages = super()._build_openai_messages(system_text, contents)
        if 'CAPABILITY_EVIDENCE_OPTIONS' in system_text or 'You are inside Stage 2' in system_text:
            return messages
        if self._goal_contract is None:
            rule = 'The requirements field must enumerate all final task effects, not just the next prerequisite. Preserve exact scope and requested increments.'
        else:
            rule = ('Keep the main decision JSON schema and ADD goal_checks with exactly these goal IDs: '
                    + json.dumps(list(self._goal_contract)) + '. Each value must contain status '
                    '(confirmed or unconfirmed) and observed_evidence. Immutable goal definitions: '
                    + json.dumps(self._goal_contract) + '. Local prerequisites cannot confirm final effects.')
        return messages + [{'role':'user','content':rule}]

    def call_llm(self, system_text, contents):
        if 'CAPABILITY_EVIDENCE_OPTIONS' in system_text or 'You are inside Stage 2' in system_text:
            return super().call_llm(system_text=system_text, contents=contents)
        if self._goal_contract is None:
            rule = ('On this first decision, requirements must describe ALL final task effects, '
                    'objects, exact scope and requested change relative to the initial state. '
                    'Do not replace final effects with prerequisites such as selecting a region '
                    'or opening a dialog. Preserve quantities, target ranges and persistence constraints.')
        else:
            rule = ('Immutable task effects: ' + json.dumps(self._goal_contract) + '''
Local requirements may describe the current support need, but cannot replace
the immutable task effects. Add goal_checks to the JSON object: for EVERY goal
ID provide {"status":"confirmed or unconfirmed", "observed_evidence":"current evidence"}.
Selection, focus, an intended action, a reference image or a local planner note
does not confirm an editing effect. Verify requested changes relative to the
initial target state, including exact scope and increment, not absolute appearance.
DONE is allowed only when all immutable effects are confirmed. Otherwise act,
seek applicable support, or report FAIL if the goal cannot be achieved.''')
        response = super().call_llm(system_text=system_text + '\n' + rule,
            contents=contents + [{'role':'user','parts':[{'text':rule}]}])
        path = Path(self._result_dir) / 'capability_decisions.jsonl'
        decision = json.loads(path.read_text().splitlines()[-1])
        first = self._goal_contract is None
        if first:
            if not decision['requirements']:
                raise ValueError('Initial task-effect contract is empty')
            self._goal_contract = {f'G{i}':{**{k:r[k] for k in ('effect','object','scope')},
                                          'initial_observed_evidence':r['observed_evidence']}
                                   for i,r in enumerate(decision['requirements'],1)}
            (Path(self._result_dir) / 'task_effect_contract.json').write_text(
                json.dumps(self._goal_contract, indent=2) + '\n')
        elif set(decision['goal_checks']) != set(self._goal_contract):
            raise ValueError('Goal checks must cover the immutable task-effect contract')
        if response.strip().strip('`').strip().removeprefix('python').strip() == 'DONE':
            checks = decision['requirements'] if first else decision['goal_checks'].values()
            if any(r['status'] != 'confirmed' or not r['observed_evidence'] for r in checks):
                raise ValueError('DONE without evidence for all task effects')
        return response
