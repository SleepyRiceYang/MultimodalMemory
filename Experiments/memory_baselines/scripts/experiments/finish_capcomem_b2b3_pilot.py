"""Finish exactly two registered pilot iterations after a verified existing first run."""
import argparse,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--first-pid',type=int,required=True);args=p.parse_args()
root=Path('/root/VLM-Memory');os.chdir(root)
run=root/'Experiments/memory_baselines/artifacts/capcomem_rq1_b2b3'
lock=run/'finisher.lock'
import fcntl
with lock.open('w') as guard:
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    proc=Path(f'/proc/{args.first_pid}/cmdline')
    while proc.exists() and b'run_capcomem_b2b3_pilot.sh' in proc.read_bytes():time.sleep(15)
    env={**os.environ,'PYTHONPATH':'Methods/MAGNET/src'};python='/root/magnet_cold_env/bin/python'
    for condition in ['B2','B3']:
        result=json.loads((run/'iteration_1'/condition/'summary.json').read_text())
        assert result['n']==8,'First run incomplete; do not start iteration two'
    for iteration in [1,2]:
        base=run/f'iteration_{iteration}'
        if iteration==2:
            base.mkdir(exist_ok=True)
            with (base/'run.log').open('ab') as log:
                subprocess.run(['bash','Experiments/memory_baselines/scripts/experiments/run_capcomem_b2b3_pilot_v2.sh'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([python,'-m','gui_baseline.evidence_audit','--run',str(base)],env=env,check=True)
        assert json.loads((base/'audit.json').read_text())['complete']
        out=root/f'Docs/实验分析/assets/20261118-capcomem-b2b3-iteration{iteration}'
        subprocess.run([python,'-m','gui_baseline.evidence_report','--run',str(base),'--output',str(out)],env=env,check=True)
    report=root/'Docs/实验分析/20261118-02-CapCoMem-RQ1-B2B3小范围实验设置与结果.md'
    text=report.read_text();text=text.replace('状态：第一轮执行中，尚未形成效果结论。','状态：两轮均已执行完成；以下给出实际结果。')
    lines=['\n## 两轮实际结果\n','每轮固定八题，每个条件独立执行一次；两轮仅增强角色与 few-shot 提示，记忆库、任务集合与预算保持不变。\n','|轮次|B2 成功|B3 成功|B3 新增成功|B3 退化|','|---|---:|---:|---|---|']
    allresults={}
    for i in [1,2]:
        folder=root/f'Docs/实验分析/assets/20261118-capcomem-b2b3-iteration{i}'
        stats=json.loads((folder/'statistics.json').read_text());comp=json.loads((folder/'comparison.json').read_text())
        assert comp['complete'];allresults[str(i)]={'statistics':stats,'comparison':comp}
        lines.append(f'|{i}|{stats["B2"]["successes"]}/8|{stats["B3"]["successes"]}/8|'+(', '.join(comp['B3_gain_tasks']) or '无')+'|'+(', '.join(comp['B3_loss_tasks']) or '无')+'|')
    lines+=['\n上表是实际结果，不将完成执行等同于方法有效。小样本、单次执行及逐步状态分歧限制了图片作用的因果解释。后续逐题机制分析以完整轨迹为准。',
            '\n[第一轮可视化轨迹](assets/20261118-capcomem-b2b3-iteration1/index.html)；[第二轮可视化轨迹](assets/20261118-capcomem-b2b3-iteration2/index.html)。每页包含截图、实际选择、请求提示和模块输出。',
            '\n两轮输入审计均已通过：B2 实际 Planner/Actor 历史图像为零；B3 图片路径与所选证据一致；两组冻结记忆、任务、预算、有效选择提示一致。成本统计保存在各可视化目录的 costs.json。']
    report.write_text(text+'\n'.join(lines)+'\n')
    (run/'two_iterations_finished.json').write_text(json.dumps(allresults,ensure_ascii=False,indent=2))
    print('TWO ITERATIONS FINISHED; results exported. Manual per-task interpretation still required.',flush=True)
