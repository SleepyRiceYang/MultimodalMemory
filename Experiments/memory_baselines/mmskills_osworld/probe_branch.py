"""Exercise native branches on a saved screenshot; diagnostic only, no GUI actions."""
import argparse
import json
import logging
from pathlib import Path

import run as protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--condition', choices=('text_skill', 'visual_skill'), required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8166/v1')
    parser.add_argument('--capability-mode', choices=('text', 'cap'))
    parser.add_argument('--sidecars', type=Path)
    parser.add_argument('--force-first-view', action='store_true',
                        help='Diagnostic fixture: force one valid Stage-1 request, run native Stage 2 on the actual images')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh diagnostic output directory')
    args.output.mkdir(parents=True)
    task, domain = args.episode.name, args.episode.parent.name
    config = protocol.read(protocol.OSWORLD / 'evaluation_examples/examples' / domain / (task + '.json'))
    options = dict(base_url=args.base_url, model='qwen3-vl-8b-instruct',
                   max_tokens=65536, context_window=65536, request_timeout=300,
                   skills_dir=str(protocol.ROOT / 'Datasets/MMSkills/ubuntu'))
    if args.capability_mode:
        if args.condition != 'visual_skill' or args.sidecars is None:
            parser.error('Capability probe requires visual_skill and --sidecars')
        from capability_agent import CapabilitySkillAgent
        agent = CapabilitySkillAgent(model=options['model'], api_model=options['model'],
            base_url=options['base_url'], api_key='EMPTY', max_tokens=65536,
            temperature=0, top_p=.9, action_space='pyautogui', observation_type='screenshot',
            coordinate_type='relative', max_trajectory_length=3,
            skills_library_dir=options['skills_dir'], skill_mode='multimodal',
            capability_dir=args.sidecars.resolve(), capability_mode=args.capability_mode)
    else:
        agent = protocol.make_agent(args.condition, options)
    transport = protocol.RecordedTransport(options, args.output)
    skills = protocol.read(args.episode / 'selection.json')['selected_skills']
    forced = None
    if args.force_first_view:
        if args.condition != 'visual_skill':
            raise ValueError('Forced view diagnostic requires visual_skill')
        manifest = next(Path(options['skills_dir']).glob('*/' + skills[0] + '/runtime_state_cards.json'))
        state = protocol.read(manifest)['states'][0]
        view = next(v['view_type'] for v in state['available_views'] if v['view_type'] in ('full_frame', 'focus_crop'))
        forced = {'visual_reference_needed': True,
                  'why_not_text_only': 'DIAGNOSTIC FIXTURE: exercise actual reference-image loading and Stage 2 transport.',
                  'requests': [{'state_id': state['state_id'], 'views': [view], 'evidence_goal': 'locate_control',
                                'reason': 'DIAGNOSTIC FIXTURE: verify a valid available reference view can be loaded.'}]}
        protocol.write(args.output / 'forced_stage1_fixture.json', forced)
    def call(system_text, contents):
        if forced is not None and 'You are inside Stage 1' in system_text:
            return '```python\nLOAD_STATE_VIEWS(' + json.dumps(forced) + ')\n```'
        result = transport.complete(model=options['model'], temperature=0, top_p=.9,
            messages=agent._build_openai_messages(system_text, contents))
        return agent._extract_openai_message_text(result.choices[0].message)
    agent._call_llm_openai = call
    agent.reset(logging.getLogger('branch_probe'), result_dir=str(args.output))
    if not skills:
        raise ValueError('Source episode has no candidate skills')
    agent.set_task_skills(skills)
    from mm_agents.general_agent import process_image
    screenshot = (args.episode / 'step_00_before.png').read_bytes()
    image, width, height = process_image(screenshot)
    extra = {}
    if args.condition == 'visual_skill':
        from PIL import Image
        from io import BytesIO
        original_width, original_height = Image.open(BytesIO(screenshot)).size
        extra = dict(original_width=original_width, original_height=original_height,
                     main_trigger_response=f'LOAD_SKILL("{skills[0]}")')
    result = agent._run_skill_branch(instruction=config['instruction'],
        obs={'screenshot': screenshot, 'env_feedback': 'None'}, processed_image=image,
        processed_width=width, processed_height=height, trigger_skill_name=skills[0], step_idx=0, **extra)
    protocol.write(args.output / 'probe.json', {'diagnostic_only': True,
        'forced_consultation': True, 'gui_actions_executed': False, 'source_episode': str(args.episode),
        'forced_stage1_fixture': args.force_first_view,
        'condition': args.condition, 'capability_mode': args.capability_mode,
        'result': result, 'model_calls': len(transport.calls)})
    if not result['success']:
        raise SystemExit('Native branch did not return a valid planner summary')
    print(args.output / 'probe.json')


if __name__ == '__main__':
    main()
