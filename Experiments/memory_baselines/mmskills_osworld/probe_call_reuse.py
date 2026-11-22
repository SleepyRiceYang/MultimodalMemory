"""Local-model fixed-state probe after recorded guidance; executes no GUI actions."""
import argparse
import json
import logging
from pathlib import Path

import run as protocol
from capability_v3_agent import CapabilityV3Agent
from mm_agents.general_agent import process_image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sidecars', type=Path, required=True)
    p.add_argument('--mode', choices=['cap', 'text'], default='cap')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    options = dict(model='qwen3-vl-8b-instruct', base_url='http://127.0.0.1:8170/v1',
                   max_tokens=65536, context_window=65536, request_timeout=300)
    agent = CapabilityV3Agent(model=options['model'], api_model=options['model'],
        base_url=options['base_url'], api_key='EMPTY', max_tokens=65536,
        temperature=0, top_p=.9, skills_library_dir=str(protocol.ROOT / 'Datasets/MMSkills/ubuntu'),
        skill_mode='multimodal', capability_dir=a.sidecars.resolve(), capability_mode=a.mode)
    agent.reset(logging.getLogger('reuse_probe'), result_dir=str(a.output.resolve()))
    skills = protocol.read(a.episode / 'selection.json')['selected_skills']
    agent.set_task_skills(skills)
    invocation = protocol.read(a.episode / 'skill_invocations.json')['invocations'][0]
    agent._current_step_planner_summaries = [invocation['final_summary']]
    agent._capability_step = 0
    agent._step_called_skills = {invocation['trigger_skill_name']}
    transport = protocol.RecordedTransport(options, a.output)
    def call(system_text, contents):
        result = transport.complete(model=options['model'], temperature=0, top_p=.9,
            messages=agent._build_openai_messages(system_text, contents))
        return agent._extract_openai_message_text(result.choices[0].message)
    agent._call_llm_openai = call
    task = protocol.read(protocol.OSWORLD / 'evaluation_examples/examples' /
                         a.episode.parent.name / (a.episode.name + '.json'))
    screenshot = (a.episode / 'step_00_before.png').read_bytes()
    image, width, height = process_image(screenshot)
    contents = agent._build_main_contents(instruction=task['instruction'], processed_image=image,
        processed_width=width, processed_height=height,
        obs={'screenshot': screenshot, 'env_feedback': 'None'})
    response = agent.call_llm(agent._build_main_system_message(task['instruction']), contents)
    protocol.write(a.output / 'probe.json', dict(diagnostic_only=True, gui_actions_executed=False,
        fixture='same GUI step after recorded guidance, no new observation',
        source_episode=str(a.episode.resolve()), mode=a.mode, response=response,
        model_calls=len(transport.calls), repeated_call='LOAD_SKILL' in response))
    print(response)


if __name__ == '__main__':
    main()
