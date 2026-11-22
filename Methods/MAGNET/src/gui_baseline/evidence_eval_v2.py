"""Iteration two: same frozen memory/tasks and execution, stronger role/few-shot prompts."""
from . import evidence_eval
from .evidence_prompts_v2 import SELECT_V2,USE_V2

if __name__=='__main__':
    evidence_eval.SELECT=SELECT_V2
    evidence_eval.USE=USE_V2
    evidence_eval.main()
