"""Targeted iteration: reuse current-step guidance before repeating a skill call."""
import json

from capability_v2_agent import CapabilityV2Agent


class CapabilityV3Agent(CapabilityV2Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._capability_step = None
        self._step_called_skills = set()

    def call_llm(self, system_text, contents):
        if 'CAPABILITY_EVIDENCE_OPTIONS' in system_text or 'You are inside Stage 2' in system_text:
            return super().call_llm(system_text=system_text, contents=contents)
        step = len(self.actions)
        if step != self._capability_step:
            self._capability_step = step
            self._step_called_skills.clear()
        callable_skills = [name for name in self._task_skill_names
                           if name not in self._step_called_skills and not self._is_skill_exhausted(name)]
        context = dict(callable_skills_this_step=callable_skills,
                       already_called_this_step=sorted(self._step_called_skills),
                       current_step_guidance=self._current_step_planner_summaries)
        system_text += '\nCurrent invocation state:\n' + json.dumps(context) + '''
Use existing guidance before seeking new support. A skill can be called once
per GUI step; later steps may call it again within the original total limit.
For call, choose only callable_skills_this_step and identify information not
already supplied by current_step_guidance. If none applies, choose direct.
Do not claim unknown scope is supplied by a prior planner note.'''
        state_instruction = 'Current invocation state: ' + json.dumps(context)
        if not callable_skills:
            state_instruction += '\nNo skill is callable THIS GUI STEP. Use the existing guidance and return decision=direct, skill=null, with a grounded action; do not repeat a call.'
        response = super().call_llm(system_text=system_text,
            contents=contents + [{'role':'user','parts':[{'text':state_instruction}]}])
        prefix, suffix = '```python\nLOAD_SKILL(', ')\n```'
        if response.startswith(prefix):
            name = json.loads(response[len(prefix):-len(suffix)])
            if name not in callable_skills:
                raise ValueError('Skill already used this step or unavailable')
            self._step_called_skills.add(name)
        return response
