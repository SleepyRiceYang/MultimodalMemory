"""Experimental prompt adapter; native runtime and parsers stay unchanged."""
import json
from pathlib import Path

from mm_agents.mm_skill_agent import MMSkillAgent


class CapabilitySkillAgent(MMSkillAgent):
    def __init__(self, *args, capability_dir, capability_mode, **kwargs):
        super().__init__(*args, **kwargs)
        self.capability_dir = Path(capability_dir)
        self.capability_mode = capability_mode

    def _capability_context(self, names):
        entries = []
        for name in names:
            row = json.loads((self.capability_dir / (name + '.json')).read_text())
            labels = row['labels']
            if self.capability_mode == 'cap':
                entries.append(json.dumps({'skill': name, **labels}, ensure_ascii=False))
            else:
                lines = [f'Skill: {name}']
                for relation in ('supports', 'requires'):
                    for item in labels[relation]:
                        lines.append(f"{'Provides' if relation == 'supports' else 'Needs'} "
                                     f"{item['capability']}: {item['effect']}; scope: {item['scope']}. "
                                     f"Source {item['source_file']}: {item['quote']}")
                lines.append('Preconditions: ' + json.dumps(labels['preconditions']))
                lines.append('Unknown scope: ' + json.dumps(labels['unknown_scope']))
                for item in labels['state_evidence']:
                    lines.append(f"State {item['state_id']}, function {item['capability']}, "
                                 f"question: {item['question']}. Source {item['source_file']}: {item['quote']}")
                entries.append('\n'.join(lines))
        return '\n\n'.join(entries)

    def _guidance(self, stage, names):
        checks = {
            'main': 'Identify the current goal, object, constraints and unresolved question. '
                    'Check whether a candidate provides new useful information beyond the current screenshot and memo. '
                    'Call only when useful; otherwise act directly. Do not treat prerequisites as provided support.',
            'views': 'Identify the unresolved state question. Choose only available states/views that answer it. '
                     'If text and the current screenshot suffice, load no reference images.',
            'planner': 'Identify the local target effect, check prerequisites, adapt objects and parameters to the current task, '
                       'and state how to verify the effect in the CURRENT environment. Do not copy example coordinates.'}
        relation = ('Explicitly match Task requires to Skill supports, check Skill requires, and use state_evidence '
                    'to answer the remaining question. ' if self.capability_mode == 'cap' else
                    'Reason through the supplied functional notes and their applicability to the current task. ')
        return '\n\nAdditional functional reference (not proof of current completion):\n' + \
            self._capability_context(names) + '\n' + relation + checks[stage] + \
            '\nKeep the original response format and all original action/call/view limits.'

    def _build_main_system_message(self, instruction):
        return super()._build_main_system_message(instruction) + self._guidance('main', self._task_skill_names)

    def _build_branch_view_selection_system_message(self, instruction, loaded_skill_names):
        return super()._build_branch_view_selection_system_message(instruction, loaded_skill_names) + \
            self._guidance('views', loaded_skill_names)

    def _build_branch_planner_system_message(self, instruction, loaded_skill_names):
        return super()._build_branch_planner_system_message(instruction, loaded_skill_names) + \
            self._guidance('planner', loaded_skill_names)
