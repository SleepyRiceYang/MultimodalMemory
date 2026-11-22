"""Summarize a completed frozen KSI retrieval pilot from saved artifacts."""
import argparse
import json
from collections import Counter
from pathlib import Path
from ..config import ROOT, WORKSPACE, write_json


def read(path):
    return json.loads(path.read_text())


def report(root):
    manifest = read(root/'dataset_manifest.json')
    config = read(root/'config.snapshot.json')
    validation = read(root/'validation.json')
    if not validation['all_completed'] or validation['completed_results'] != 10:
        raise ValueError('Report requires all 5 paired results')
    bank = read(root/'bank.json')
    nodes = bank['methods']['ksi_style']
    results = {method: [read(root/'runs'/method/t['id']/'seed_0/attempt_0/result.json') for t in manifest['test']] for method in ['no_memory','ksi_style']}
    traces = [read(root/'runs/ksi_style'/t['id']/'seed_0/attempt_0/memory_trace.json') for t in manifest['test']]
    covered = {s for n in nodes for s in n['source_ids']}
    source = ROOT/'artifacts/vision2code'/config['source_run']
    constructions = [read(source/'runs/no_memory/memory'/t['id']/'attempt_0/result.json') for t in manifest['memory']]
    summary = {method: {'successes':sum(bool(r.get('success')) for r in rows),'mean_score':sum(float(r.get('score') or 0) for r in rows)/len(rows),'renders':sum(bool(r['evaluation'].get('render_success')) for r in rows)} for method,rows in results.items()}
    selected = [len(t['selected']) for t in traces]
    query_failures = sum(bool(t['retrieval_metadata'].get('query',{}).get('query_rewrite_failed')) for t in traces)
    calls = []
    for p in root.rglob('*.json'):
        if 'no_memory' in p.relative_to(root).parts:
            continue
        if 'calls' not in p.parts and 'construction_calls' not in p.parts:
            continue
        if p.name.endswith(('.partial.json','.request.json')):
            continue
        x = read(p)
        if isinstance(x,dict) and 'usage' in x and 'finish_reason' in x:
            calls.append((p,x))
    for task in manifest['test']:
        path = root/'runs/ksi_style'/task['id']/'seed_0/attempt_0/evaluation.json'
        evaluation = read(path)
        ratings = evaluation.get('rater_calls', [])
        for rating in ratings:
            calls.append((path, {**rating, 'seconds': evaluation.get('rater_seconds', 0)/max(len(ratings),1)}))
    usage = Counter()
    for _,call in calls:
        for key in ['prompt_tokens','completion_tokens','total_tokens']:
            usage[key] += call.get('usage',{}).get(key,0) or 0
    grades = Counter(row['grade'] for trace in traces for row in trace['retrieval_metadata'].get('candidates',[]))
    details = {'summary':summary,'selected_counts':selected,'query_failures':query_failures,'grades':dict(grades),'knowledge_nodes':len(nodes),'source_coverage':len(covered),'new_model_calls':len(calls),'new_call_tokens':dict(usage),'new_call_seconds_sum':sum(c['seconds'] for _,c in calls),'new_length_finishes':[str(p.relative_to(root)) for p,c in calls if c['finish_reason']=='length']}
    write_json(root/'retrieval_summary.json',details)
    lines = ['# Vision2Code：KSI 5/5 统一向量检索小测', '', '日期：2026-10-24。实验目的：验证新检索链路能完成建库、冻结、检索、代码生成与评分；本次样本量不足以证明方法优劣。', '',
             f"运行目录：[{root.name}]({root})。配置：[qwen3_vl_8b_ksi_vision2code_5_5.json]({ROOT/'config/qwen3_vl_8b_ksi_vision2code_5_5.json'})。", '',
             '## 设置与数据', '',
             '- 模型：本地 Qwen3-VL-8B-Instruct，GPU 5 / 8079；输出上限 8,192，记忆构建上限 32,768，temperature=0。',
             '- Embedding：本地 BAAI-bge-large-en-v1.5，GPU 7 / 8087；CLS pooling、L2 归一化、1,024 维。query 加 BGE 检索指令前缀，索引不加。',
             '- 构建集 5 题、测试集 5 题，来自既有独立 100/100 清单。种子 20261024，在五个最大类别中各抽一题，不按成功率选择。未包含 Spatial，不能视为全量分布代表。',
             '- 复用同一模型与当前基础提示词的已有 no-memory 构建轨迹，以及这五道测试题的 no-memory 对照；重新执行 KSI 记忆提炼、索引、检索和测试。测试反馈未进入记忆构建。',
             '- KSI-style 是现有项目适配实现；新检索借鉴 EPICON 的字段和两阶段结构，不等同于其训练后 organizer，也不是 KSI 原生检索复现。', '',
             '## 检索实现', '',
             '节点保留原有知识内容，另建 Topic / Scope / Question Anchor 索引。索引上限 384 个 encoder tokens，query 上限 192（含指令前缀及特殊 token）；超限显式压缩一次，编码服务拒绝超过 512 tokens 的输入。当前图像参与 query 改写；改写仅用于检索，不直接加入解题提示词。', '',
             '以归一化向量点积精确召回 Top-20；不足 20 时取全部。本次只有 6 个节点，因此每题召回全部节点；该小测不能验证大库中的召回优势。同一 Qwen 模型进行列表重排，按 0–3 级适用性、scope 冲突与重复项过滤，再按等级、cosine、ID 排序。至多注入 3 条、16,000 字符、2 张历史图像；预算不足跳过整条并继续填充。', '',
             '三个 benchmark 接入同一机制，各自独立索引。多页文档的检索描述固定读取前两页，不使用答案页；本次仅实际验证 Vision2Code。旧配置继续使用 BM25，新协议通过 retrieval.enabled=true 显式启用。', '',
             '## 结果', '', '| 方法 | 成功 / 5 | 平均分 / 5 | 渲染成功 / 5 |', '|---|---:|---:|---:|']
    for method,label in [('no_memory','No Memory（已有对照）'),('ksi_style','KSI + 新检索')]:
        x=summary[method];lines.append(f"| {label} | {x['successes']}/5 | {x['mean_score']:.3f} | {x['renders']}/5 |")
    lines += ['', '成功定义：渲染成功且评分至少 3/5。评分沿用项目 Vision2Code rubric 与本地模型评价器，非人工标注；使用相同模型生成与评分存在评价偏差。', '', '| 类别 / task ID | No Memory | KSI | 注入条数 |', '|---|---:|---:|---:|']
    for i,task in enumerate(manifest['test']):
        lines.append(f"| {task['metadata']['category']} / `{task['id']}` | {float(results['no_memory'][i].get('score') or 0):.1f} | {float(results['ksi_style'][i].get('score') or 0):.1f} | {selected[i]} |")
    lines += ['', '## 建库与检索诊断', '', f"- 五个构建任务的 no-memory 成功 {sum(bool(r.get('success')) for r in constructions)}/5，平均分 {sum(float(r.get('score') or 0) for r in constructions)/5:.3f}/5。", f'- 最终知识节点 {len(nodes)} 条，覆盖 {len(covered)}/5 个构建任务。',f'- 五题注入数量：{selected}；空检索 {sum(n==0 for n in selected)}/5；query 改写失败 {query_failures}/5。',f'- 候选适用性等级计数：{dict(sorted(grades.items()))}。',f"- 新执行模型调用 {len(calls)} 次，总输入 tokens {usage['prompt_tokens']:,}、输出 tokens {usage['completion_tokens']:,}；调用耗时之和 {details['new_call_seconds_sum']:.1f} 秒（不含复用 no-memory 开销，非端到端墙钟时间）。", f"- 新调用触及长度上限：{len(details['new_length_finishes'])} 次；测试完成 {validation['completed_results']}/{validation['expected_results']} 条结果，记忆库冻结验证通过。", '', '未被最终跨任务整合保留的构建任务：']
    lines += [f"- `{t['id']}`（{t['metadata']['category']}）" for t in manifest['memory'] if t['id'] not in covered]
    lines += ['', '各节点的来源覆盖：', '', '| 节点 | 来源任务 |', '|---|---|']
    lines += [f"| `{node['id']}` | {', '.join(node['source_ids'])} |" for node in nodes]
    lines += ['', '## 解释与限制', '', '应分别观察：建库是否提炼出可泛化知识、embedding 是否召回、重排是否判断适用、以及注入后是否改善生成。本次整合后的部分知识仍包含特定题目的数值、标签和描述，且出现 claim 声称正确、exclusions 同时指出错误的表述；这是记忆内容质量问题，新检索没有改写其载荷。未覆盖的类别在建库阶段已丢失，不能归因于 embedding。', '', '空检索题不会注入历史上下文，若与已有 no-memory 得分不同，也不能解释为记忆收益；模型运行和自动评分可能存在波动。本次没有设置 BM25 配对运行、没有大库召回标注，也没有独立 Development 集，因此只能确认链路与观察个案，不能证明 embedding 优于 BM25 或 KSI 泛化能力提升。', '', '## 复现与产物', '', '```bash', 'bash Experiments/memory_baselines/scripts/serve/start_embedding.sh', 'bash Experiments/memory_baselines/scripts/serve/start_vision2code.sh', 'bash Experiments/memory_baselines/scripts/diagnostics/run_ksi_vision2code_5_5.sh', '```', '', '已有服务时不重复启动。实验命令支持复用完成调用；修改配置、提示词或划分应使用新 run ID。', '', f'- [划分清单]({root/"dataset_manifest.json"})；[汇总]({root/"retrieval_summary.json"})；[完整验证]({root/"validation.json"})。', f'- [冻结知识库]({root/"bank.json"})；[索引]({root/"retrieval/index.json"})；[检索轨迹]({root/"retrieval/traces/ksi_style"})；[运行日志]({root/"runner.log"})。', '- 实际提示词、输出、token 使用和耗时均保存在 construction_calls、retrieval/calls 和 runs 下的 calls 文件中。未复制代码版本或新增数据哈希流程。', '']
    path=WORKSPACE/'Docs/实验分析/20261024-02-Vision2Code-KSI-5构建5测试-统一向量检索.md'
    path.write_text('\n'.join(lines))
    print(path)
    return path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='qwen3_vl_8b_ksi_dense_5_5_20261024')
    args=parser.parse_args()
    report(ROOT/'artifacts/vision2code'/args.run)

if __name__=='__main__':
    main()
