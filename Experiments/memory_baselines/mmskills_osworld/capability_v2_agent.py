"""Capability decisions control native invocation; branch/action runtime is retained."""
import json
import re
from pathlib import Path

from capability_agent import CapabilitySkillAgent


class CapabilityV2Agent(CapabilitySkillAgent):
    def _build_main_system_message(self, instruction):
        prompt = super()._build_main_system_message(instruction)
        start = prompt.index('Output format:')
        end = prompt.index('Additional functional reference')
        return prompt[:start] + 'Output follows the structured main-decision schema below.\n' + prompt[end:]

    def call_llm(self, system_text, contents):
        if 'CAPABILITY_EVIDENCE_OPTIONS' in system_text:
            rule = 'Return JSON only: {"why":"current capability need and evidence gap, or why text suffices", "selections":[{"option_id":"exact listed ID", "reason":"capability need and what this evidence answers"}]}. Use [] if no image adds useful evidence. At most 3 selections, one per state.'
            raw = super().call_llm(system_text=system_text + '\n' + rule,
                contents=contents + [{'role':'user','parts':[{'text':rule}]}])
            decision = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip()))
            requests = []
            for item in decision['selections']:
                requests.append({**self._evidence_options[item['option_id']], 'reason':item['reason']})
            if len(requests) > 3 or len({r['state_id'] for r in requests}) != len(requests):
                raise ValueError('Evidence selection exceeds state limit or repeats a state')
            payload = dict(visual_reference_needed=bool(requests), why_not_text_only=decision['why'], requests=requests)
            with (Path(self._result_dir) / 'capability_evidence_decisions.jsonl').open('a') as stream:
                stream.write(json.dumps(decision) + '\n')
            return '```python\nLOAD_STATE_VIEWS(' + json.dumps(payload) + ')\n```'
        if 'You are inside Stage 2' in system_text:
            return super().call_llm(system_text=system_text, contents=contents)
        instruction = '''
For this MAIN decision, replace the earlier output format with one JSON object:
{"requirements":[{"effect":"...","object":"...","scope":"...",
"observed_evidence":"...","status":"confirmed or unconfirmed"}],
"support_need":"specific unresolved question, or none",
"skill_support":"what a listed skill provides for that question, or unknown",
"decision":"call or direct","skill":null,"reason":"...","action":"..."}.
For call: skill is an exact available non-exhausted skill name, action is empty.
For direct: skill is null; action contains the original single Python code block,
WAIT, DONE or FAIL. Do not put LOAD_SKILL in action.
Use current evidence, not prior action intent, to confirm effects.
Text content, object structure and persistence are separate effects.
Call only if a listed skill can address the current unresolved need; a shared
capability label alone is insufficient. Unknown scope is not support.
Return JSON only, without an outer markdown fence.'''
        raw = super().call_llm(system_text=system_text + instruction,
            contents=contents + [{'role':'user','parts':[{'text':instruction}]}])
        decision = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip()))
        if decision['decision'] not in ('call', 'direct'):
            raise ValueError('Invalid invocation decision')
        if not isinstance(decision['requirements'], list):
            raise ValueError('Requirements must be a list')
        for item in decision['requirements']:
            if item['status'] not in ('confirmed', 'unconfirmed'):
                raise ValueError('Invalid effect confirmation status')
        with (Path(self._result_dir) / 'capability_decisions.jsonl').open('a') as stream:
            stream.write(json.dumps(decision, ensure_ascii=False) + '\n')
        if decision['decision'] == 'call':
            skill = decision['skill']
            if skill not in self._task_skill_names or self._is_skill_exhausted(skill):
                raise ValueError('Decision requests an unavailable skill')
            return '```python\nLOAD_SKILL(' + json.dumps(skill) + ')\n```'
        if decision['skill'] is not None or 'LOAD_SKILL' in decision['action']:
            raise ValueError('Direct decision must not invoke a skill')
        return decision['action']

    def _build_branch_view_selection_system_message(self, instruction, loaded_skill_names):
        self._evidence_options = {}
        for name in loaded_skill_names:
            source = next(self._skill_loader._skills_dir.glob('*/' + name + '/runtime_state_cards.json'))
            for state in json.loads(source.read_text())['states']:
                for view in state['available_views']:
                    kind = view['view_type']
                    goal = {'full_frame':'locate_control','focus_crop':'locate_control',
                            'before':'recognize_before','after':'verify_after'}[kind]
                    key = f'{name}/{state["state_id"]}/{kind}'
                    self._evidence_options[key] = dict(state_id=state['state_id'],views=[kind],evidence_goal=goal)
                kinds = {v['view_type'] for v in state['available_views']}
                if {'before','after'} <= kinds:
                    self._evidence_options[f'{name}/{state["state_id"]}/transition'] = dict(
                        state_id=state['state_id'], views=['before','after'], evidence_goal='compare_transition')
        return 'CAPABILITY_EVIDENCE_OPTIONS\nSelect evidence for the CURRENT capability gap from this exact catalog. Historical images do not establish current completion. Preserve task scope and card applicability; choose no images when text suffices.\n' + self._capability_context(loaded_skill_names) + '\n' + json.dumps(self._evidence_options)

    def _build_branch_planner_system_message(self, instruction, loaded_skill_names):
        return super()._build_branch_planner_system_message(instruction, loaded_skill_names) + '''
In plan, distinguish the effects supported by skill materials from unknown
scope or model-proposed alternatives. Do not invent a menu path and attribute
it to the skill. In expected_state, separately state current evidence needed
for content, structure and persistence whenever those effects are requested.
An intended action or historical reference image never proves completion.'''
