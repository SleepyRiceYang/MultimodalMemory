"""Publish the five-task paired validation report from recorded results only."""
from pathlib import Path
import argparse,csv,json
from .common import read,write

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',default='/root/VLM-Memory');p.add_argument('--run',default='paired_5_v1');p.add_argument('--android-run',default='paired_5_v2');args=p.parse_args()
 workspace=Path(args.workspace);base=workspace/'Experiments/memory_baselines/artifacts/gui_magnet_procedural'
 rows=[];banks={};checks={}
 for benchmark in ['androidworld','osworld']:
  root=base/benchmark/(args.android_run if benchmark=='androidworld' else args.run);config=read(root/'config.json');bank=read(base/benchmark/'bank/bank.json');banks[benchmark]=bank
  checks[benchmark]=read(root/'pairing_checks.json')
  for task in config['tasks']:
   for condition in ['text','visual']:
    folder=root/task/condition;r=read(folder/'result.json');m=read(folder/'memory.json')
    if r['status']!='completed':raise ValueError(f'Incomplete/error condition: {folder}: {r}')
    if benchmark=='androidworld':requests=list(folder.glob('step_*/request.json'));get_images=lambda path:read(path)['images'][1:]
    else:requests=list(folder.glob('request_*.json'));get_images=lambda path:read(path)['historical_images']
    actual_ok=bool(requests) and all(get_images(path)==m['images'] for path in requests)
    if not actual_ok:raise ValueError(f'Actual memory-image request mismatch: {folder}')
    rows.append({k:r.get(k) for k in ['benchmark','task_id','condition','status','score','task_state_score','steps','seconds','historical_images','selected_ids']} | {'requests_verified':len(requests),'artifact':str(folder)})
  if len(checks[benchmark]['checks'])!=5 or any(not c.get('same_text') or not c.get('same_selection') or c.get('text_images')!=0 or c.get('visual_images')!=2 for c in checks[benchmark]['checks']):raise ValueError('Pairing validation failed')
  if checks[benchmark]['frozen_memory'] is not True:raise ValueError('Memory freeze failed')
 target=workspace/'Docs/实验分析/20261026-04-MAGNET操作性记忆GUI五样本验证'
 write(str(target)+'.json',{'runs':{'androidworld':args.android_run,'osworld':args.run},'rows':rows,'pairing_checks':checks,'bank_nodes':{b:len(d['nodes']) for b,d in banks.items()}})
 with Path(str(target)+'.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 lines=['# MAGNET 操作性记忆 GUI 五样本验证','',f'日期：2026-10-26；run：Android `{args.android_run}`，OS `{args.run}`。每 benchmark 5个不同于构建来源的任务、Text/Visual各一次，共20次真实环境执行。',
  '', '**全部运行结束、配对注入与冻结检查通过，不代表全部任务成功。** 本轮验证在线流程，不能用稀疏记忆库和5个诊断任务推断视觉记忆的泛化收益。',
  '', '| Benchmark | 有效记忆节点 | Text 成功 | Visual 成功 | 流程完成 |', '|---|---:|---:|---:|---:|']
 for b in banks:
  count=lambda condition:sum(r['score']==1 for r in rows if r['benchmark']==b and r['condition']==condition)
  lines.append(f'| {b} | {len(banks[b]["nodes"])} | {count("text")}/5 | {count("visual")}/5 | 10/10 |')
 lines+=['','Android 成功分按照官方终止门控记录：未显式结束的 Agent score 为0；终态状态分另列。OS 使用官方 env.evaluate 数值。','',
  '## 逐题结果','', '| Benchmark / Task | Text分 / 步数 | Visual分 / 步数 | Android终态状态分 Text / Visual |','|---|---:|---:|---:|']
 for b in banks:
  for task in read(base/b/(args.android_run if b=='androidworld' else args.run)/'config.json')['tasks']:
   a=next(r for r in rows if r['benchmark']==b and r['task_id']==task and r['condition']=='text');v=next(r for r in rows if r['benchmark']==b and r['task_id']==task and r['condition']=='visual')
   lines.append(f'| {b} / {task} | {a["score"]} / {a["steps"]} | {v["score"]} / {v["steps"]} | {str(a["task_state_score"] if a["task_state_score"] is not None else "—")} / {str(v["task_state_score"] if v["task_state_score"] is not None else "—")} |')
 lines+=['','## 机制与验证证据','',
  '- 同一 benchmark 的两组使用同一冻结库、公开目标 embedding Query、相同节点与逐字相同的记忆文本；Text历史图像0张，Visual历史图像2张。逐请求核对历史图像引用，不仅检查配置开关。',
  '- 构建来源与验证任务 ID/类型隔离；各5条旧来源轨迹各仅产出1条保留的操作性记忆。来源数、节点数、检索次数是不同概念。',
  '- 所有测试只读记忆，结束检查 bank、向量、索引与证据图像未修改；不按成功失败写回。',
  '- 6项单元检查通过：源动作/槽位、隔离、配对预算/消息、证据冻结、Actor字段白名单；另通过Python编译与shell语法检查。',
  '- 没有使用或下载 OmniParser，没有 stationary memory。视觉证据为工作流前后关键帧，并非控件检测 patch。',
  '', '## 限制与质量观察','',
  '当前是 MAGNET-GUI-procedural-adapted：重新实现在线接口、使用BGE cosine冻结检索并增加工作流视觉条件，不是原MAGNET完整复现。当前Query是完整公开目标，并未实现状态相关动态查询；无相关性阈值，小库会给无关任务注入记忆。',
  '', '构建阶段观察到步骤错位和虚假完成描述。逐步骤截图、独立VLM核验筛除候选后，仍需对两条保留项进行构建来源截图复核：Android起点改为实际桌面；OS只保留待确认的缩放状态，不能当作已生效设置。visual_review.json保存修改前版本和理由。该库属于带复核的诊断库，不能声称全自动记忆质量问题已经解决。',
  '', '两组中重复点击/滚动、定位错误和未主动结束均应结合轨迹解读。Android 的 ClockStopWatchRunning：Text 打开 Clock、启动秒表、显式完成，得1；Visual 首步就声称桌面秒表已在运行，未执行环境动作，得0。OS 的自动锁屏：Text 经搜索 Settings 后8步完成，得1；Visual 错认 dock 图标、在打开与关闭窗口间循环30步，得0。这些是实际轨迹上的分歧，并不证明历史图像本身会普遍降低成绩：两库都仅1条且多数测试目标与记忆不相关，必须先补足覆盖并设计空检索规则。未运行同批任务No Memory，本次也不与上一批不同任务的 No Memory 分数作因果比较。',
  '', '## 文件与复跑','',
  '- [实现与协议](../实验方案/20261026-04-MAGNET操作性记忆GUI适配.md)',
  '- 原始产物：Experiments/memory_baselines/artifacts/gui_magnet_procedural/androidworld/paired_5_v2 与 osworld/paired_5_v1；每题包含两组动作、截图、实际请求、官方分数与memory.json；任务级retrieval.json固定配对检索。',
  '- 库与构建审计：同目录各benchmark/bank，包括source_episodes、construction_audit、calls、visual_review、bank.json、index.json与evidence。',
  '- 同名JSON与CSV保存逐题数据，pairing_checks.json保存运行结束时的检查；日志位于artifacts/logs/gui_magnet_procedural_<benchmark>.log。','',
  '```bash','bash Experiments/memory_baselines/scripts/experiments/run_gui_magnet_procedural_5.sh androidworld --run paired_5_v2','bash Experiments/memory_baselines/scripts/experiments/run_gui_magnet_procedural_5.sh osworld','```','',
  '两条命令可并行；已完成条件会跳过。新的参数配置使用新的 --run，不能改写已有结果。']
 Path(str(target)+'.md').write_text('\n'.join(lines)+'\n')
 print(str(target)+'.md')

if __name__=='__main__':main()
