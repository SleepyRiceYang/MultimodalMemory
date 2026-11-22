"""Prompts for the procedural-only GUI adaptation; stationary memory is disabled."""
BUILD = '''Extract at most TWO reusable procedural workflows from the RECORDED GUI actions and screenshots.
This is evidence extraction, not solving a new task. Each workflow must use 2-8 consecutive recorded actions.
Prefer distinct successful state transitions. Failed overall tasks may contain useful local transitions,
but NEVER label the failed overall goal as achieved. Do not include repeated no-op actions as a workflow.
Describe the ACTUAL procedure, start condition and visible end condition. The provided numeric step IDs are authoritative: NEVER renumber or shift them. The precondition describes the first BEFORE screenshot; the postcondition describes the last AFTER screenshot. Every step cites its recorded
step ID and describes that action only. Do not invent search, verification, save or navigation operations.
Parameterize task-specific names, phone numbers, filenames and typed values with <slot_name>. Preserve
operation semantics such as on/off, max/min, send/draft. Only propose parameter values present in the
source goal or recorded action. Historical coordinates are not reusable locators: describe target meaning.
Return JSON: {"workflows":[{"title":"...","app":"...","precondition":"...","goal":"...",
"parameters":[{"name":"slot_name","value":"actual source literal"}],
"steps":[{"source_step":0,"instruction":"...","expected_state":"..."}],"postcondition":"..."}]}.
Return an empty list if no supported nontrivial transition is visible.'''
VERIFY = '''Verify a proposed PROCEDURAL memory against actual source actions and BEFORE/AFTER screenshots.
Judge the local transition, not the entire source task. Reject invented actions, no-op/repetition,
incomplete actions presented as completed, or claims not supported by the observations. A source task's
terminal state score supports its overall goal only; a failed source task needs independent visible
proof for a narrower subgoal. Check that every procedural instruction corresponds to its cited action.
You may use the source state score but have no access to hidden reference answers or evaluator code.
Return JSON {"supported":true/false,"before_evidence":"specific visible starting evidence",
"after_evidence":"specific visible changed-state evidence","reason":"..."}.
Reject if uncertain. Do not infer success just because an action was attempted.'''
MEMORY_NOTICE = '''Historical PROCEDURAL memory from different tasks follows. It is optional guidance,
not the current task's facts or a mandatory plan. Check preconditions against the CURRENT screenshot.
Bind all <slots> to CURRENT task values. Re-ground each target in the current screen. Never copy source
coordinates, UI indexes, names, dates or answers. Historical images, when supplied, are BEFORE/AFTER
workflow evidence only; they are not current screens and are not a stationary target-locator memory.
If a procedure is unrelated or its precondition is false, ignore it. Current task observations prevail.
'''
ANDROID_ACTOR = '''You operate an Android phone using the CURRENT screenshot and task.
Return one JSON object with reason (one short sentence), action_type, and exactly the fields required:
click/long_press/double_tap: x,y normalized to 0..1000; input_text: text; scroll: direction up/down/left/right;
keyboard_enter, navigate_back, navigate_home, wait: no fields; open_app: app_name;
answer: text for a question; status: goal_status complete or infeasible.
input_text operates on the focused field and the native adapter may press Enter afterwards: inspect the
next screenshot before typing again. Click to focus before typing. For sliders, target the actual track.
Use the simplest visible path. After two ineffective attempts, choose a DIFFERENT action/locator/search.
Do not press Back repeatedly on the home screen: open the required app instead.
When the requested goal is visibly satisfied, immediately output status complete. Saving/creation often
changes screen silently: inspect rather than repeat Save. Do not modify beyond the requested goal.
Use only GUI actions. You cannot inspect task evaluators, hidden answers or source files.'''
