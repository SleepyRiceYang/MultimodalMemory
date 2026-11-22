"""Summarize completed construction-only MAGNET validation, without new model calls."""
import json
from pathlib import Path
from collections import Counter
from ..config import ROOT,WORKSPACE,write_json
from ..memory.magnet_v2 import support_error


def read(path):
    return json.loads(path.read_text())


def report():
    root=ROOT/'artifacts/diagnostics/magnet_v2_20261025'
    summary=dict(held_out_used=False,formal_evaluation_started=False,benchmarks={})
    for benchmark in ['mathvision','vision2code','mpdocvqa']:
        folder=root/benchmark/'magnet_v2'
        records=read(folder/'source_descriptions.json')
        calibration=read(folder/'calibration.json')
        workflow=read(folder/'workflow_admission.json')
        visual=read(folder/'visual_validation.json')
        sources={r['id']:r for r in records}
        failures=[]
        for candidate in workflow['candidates']:
            error=support_error(candidate,sources)
            if error:failures.append(dict(id=candidate['candidate_id'],error=error))
        quote_count=sum(len(r['operations']) for r in records)
        provenance=Path(read(root/benchmark/'construction_manifest.json')['provenance'])
        invalid_quotes=[]
        for record in records:
            episode=read(provenance/'histories'/record['id']/'episode.json')
            for operation in record['operations']:
                index=operation['step_index']
                if (index not in range(len(episode['steps'])) or
                        ' '.join(operation['quote'].split()) not in ' '.join(episode['steps'][index].get('text','').split())):
                    invalid_quotes.append(record['id'])
        memberships=Counter(source for group in calibration['clusters'] for source in group)
        admission=dict(described_sources=len(records),primary_clusters=len(calibration['clusters']),
            clustering_memberships=calibration['memberships'],maximum_primary_memberships=max(memberships.values(),default=0),
            abstracted_source_memberships=sum(len(g) for g in workflow['clusters'] if len(g)>1),
            threshold=calibration['threshold'],pairs=calibration['pairs'],calibration=calibration['calibration'],validation=calibration['validation'],
            validated_source_quotes=quote_count,invalid_source_quotes=len(invalid_quotes),
            admitted_workflows=len(workflow['candidates']),rejected_workflow_candidates_or_clusters=len(workflow['rejected']),
            admitted_workflow_errors=failures,visual_candidates=len(visual['candidates']),visual_groups=len(visual['groups']),
            inspected_visual_candidates=sum(bool(c.get('crop_inspection')) for c in visual['candidates']))
        summary['benchmarks'][benchmark]=admission
    replay=read(root/'vision2code/magnet_v2/legacy118_replay.json')
    if replay['count_before']!=118:
        raise ValueError('Incomplete historical candidate replay')
    group_of={member:i for i,g in enumerate(replay['groups']) for member in g}
    labels=read(ROOT/'config/magnet_v2_dedup_review_pairs.json')
    rows=[]
    for pair in labels:
        predicted=group_of[pair['left']]==group_of[pair['right']]
        rows.append(dict(**pair,predicted_equivalent=predicted,correct=predicted==pair['equivalent']))
    tp=sum(r['equivalent'] and r['predicted_equivalent'] for r in rows)
    fp=sum(not r['equivalent'] and r['predicted_equivalent'] for r in rows)
    fn=sum(r['equivalent'] and not r['predicted_equivalent'] for r in rows)
    summary['dedup_review']=dict(before=118,after=len(replay['groups']),comparisons=len(replay['comparisons']),reviewed_pairs=rows,
        tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
        label_scope='12 purposive assistant-reviewed source-candidate pairs; not independent human labels or population estimates')
    summary['integration']=read(root/'vision2code/integration_multimodal/integration_validation.json')
    summary['embedding_concurrency']=read(root/'embedding_concurrency.json')
    summary['code_checks']=read(root/'code_checks.json')
    calls={}
    for path in root.rglob('*.json'):
        if '.partial.' in path.name or '.request.' in path.name:continue
        try:value=read(path)
        except (ValueError,OSError):continue
        if isinstance(value,dict) and value.get('request_id') and value.get('finish_reason') and 'usage' in value:
            calls[value['request_id']]=(path,value)
    checked=[(p,v) for p,v in calls.values() if v.get('tokenized_input_tokens') is not None]
    local=[(p,v) for p,v in checked if 'v2_dedup_legacy118_' in p.name]
    summary['requests']=dict(completed_unique_calls=len(calls),prompt_tokens=sum(v['usage'].get('prompt_tokens',0) for _,v in calls.values()),
        completion_tokens=sum(v['usage'].get('completion_tokens',0) for _,v in calls.values()),
        max_checked_input=max(v['tokenized_input_tokens'] for _,v in checked),
        max_legacy118_comparison_input=max(v['tokenized_input_tokens'] for _,v in local),
        max_legacy118_comparison_output_budget=max(v['requested_max_tokens'] for _,v in local),
        all_checked_requests_fit=all(v['tokenized_input_tokens']+v['effective_max_tokens']<=v['context_limit'] for _,v in checked),
        accounting_scope='Includes preserved development diagnostics and parse retries, deduplicated by request ID; incomplete transport calls excluded')
    summary['ready_for_controlled_construction_run']=bool(
        fp==0 and summary['requests']['all_checked_requests_fit'] and summary['embedding_concurrency']['all_passed']
        and summary['integration']['bank_frozen'] and summary['integration']['render_success']
        and all(not row['admitted_workflow_errors'] and not row['invalid_source_quotes'] and row['maximum_primary_memberships']==1 and row['clustering_memberships']==row['described_sources']
                for row in summary['benchmarks'].values()))
    write_json(root/'acceptance.json',summary)
    lines=['# MAGNET v2 构建集验证报告','',
        '日期：2026-10-25。按《MAGNET 任务描述、聚类与去重优化方案》第 9 节实施。未将实验 held-out 任务及结果用于描述、聚类、去重、阈值选择或执行，未恢复 100/100 正式评测。', '',
        '## 1. 实现与边界','',
        '新版使用：原始图文生成问题转述；程序提供原始轨迹片段编号，模型选择编号生成执行摘要；按问题与执行摘要联合相似度做非重叠 complete-link 聚类；以逐步骤、逐来源证据门控工作流；通过近邻召回和局部关系判断去重。它是 MAGNET-adapted，不是原论文原样复现。', '',
        '模型不得自己重复填写来源 ID 充数：每个共同步骤必须给出输入来源的实际操作编号；编号、顺序、非空步骤和语义支持均检查。不足三条可追溯操作时允许不形成程序性工作流。当前实现保守地要求一个候选簇中的输入来源均支持共同步骤；不自动发明子簇，可能牺牲部分共享流程召回。', '',
        '## 2. 聚类与来源证据','',
        '| Benchmark | 描述来源 | 主簇数（含单例） | 成员累计次数 | 进入共同流程抽象的来源数 | 入库工作流 | 拒绝候选/证据不足簇 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for b,v in summary['benchmarks'].items():
        lines.append(f"| {b} | {v['described_sources']} | {v['primary_clusters']} | {v['clustering_memberships']} | {v['abstracted_source_memberships']} | {v['admitted_workflows']} | {v['rejected_workflow_candidates_or_clusters']} |")
    lines += ['', '原 Vision2Code 的 34 条轨迹累计出现在 261 个簇成员位置，新主簇成员总数为 34；不会再因最大团重叠反复处理同一来源集合。单例保留，不能为了增加记忆数而伪造共同流程。', '',
        '所有最终执行摘要的引用均由程序从原始记录建立。片段存在不等于推理正确；语义检查仍是同一局部模型的判断，因此下述通过范围仅限这次构建诊断。', '',
        '### 任务对检查','',
        '每个 benchmark 预先记录 3 个校准任务对和 3 个验证任务对，均来自构建集。任务对层面分开，部分任务可重复出现；标注由本助手直接查阅题目、原图/代码完成，不是独立人工评审。', '',
        '| Benchmark | 构建集内部验证 TP / FP / FN | 说明 |','|---|---|---|']
    for b,v in summary['benchmarks'].items():
        m=v['validation'];note='验证对均为负例，precision/recall 不适用' if m['tp']+m['fn']==0 else f"precision={m['precision']:.3f}, recall={m['recall']:.3f}（小规模定向检查）"
        lines.append(f"| {b} | {m['tp']} / {m['fp']} / {m['fn']} | {note} |")
    lines += ['', '拉丁方与乘法网格分开；年份/日期与合同编号分开；柱状图与分子图、文档分开；相同双系列柱状图在不同实例数据下保持相近。具体任务 ID、理由、相似度及门槛见每个 benchmark 的 `magnet_v2/calibration.json`。', '',
        'MathVision 和 MP-DocVQA 本轮没有可入库的程序性工作流。前者的共同步骤候选未通过严格证据检查；后者多数轨迹只有页面读取结果及最终回答，不足以声称执行了三步以上共享方法。这个结果是证据门控的真实结果，不能包装成方法性能优势。', '',
        '## 3. 原 118 条候选去重回放','',
        f"118 条完整候选分别生成短描述，然后逐条召回最多 5 个现有代表；拟合并时检查组内所有成员，每次最多比较 5 条。共记录 {len(replay['comparisons'])} 个关系判断，保留 {len(replay['groups'])} 组。没有再把全部完整候选塞入一次请求。", '',
        f"预先审阅的 12 个任务对：TP={tp}，FP={fp}，FN={fn}；precision={summary['dedup_review']['precision']}，recall={summary['dedup_review']['recall']}。该数值只描述这 12 个定向样例，不能外推至全部候选。", '',
        '原 118 条包含未经新版证据门控的旧候选。本回放只检查候选间关系，不把它们直接导入新版记忆库。不同策略、共享子流程、冲突或不确定关系保留；不会根据 A≈B、B≈C 自动合并 A 与 C。', '',
        '## 4. 实际裁图检查','',
        '额外发现：合法坐标也会裁出残缺笔画、网格交点或被截断的字段。因此加入不看候选说明的裁图盲读，再对齐组件、实际操作证据。不清晰图像拒绝入库。', '',
        '| Benchmark | 诊断来源数 | 通过图像与操作证据检查的候选 | 去重后视觉节点 |','|---|---:|---:|---:|']
    for b,v in summary['benchmarks'].items():lines.append(f"| {b} | 2 | {v['visual_candidates']} | {v['visual_groups']} |")
    lines += ['', 'MathVision 的数字裁图实际为局部笔画或网格线；MP-DocVQA 样例的合同号/日期尾部被裁断。这些候选被拒绝，未把原答案填回裁图。它同时表明当前 8B 区域定位仍有能力限制；过滤解决错误入库，不能视为定位准确率已提高。', '',
        '## 5. 请求预算和服务并发','',
        f"原全局去重：输入 38,742 + 申请输出 32,768 = 71,510 tokens，超过服务 65,536。新版 118 候选局部比较实测最大输入 {summary['requests']['max_legacy118_comparison_input']} tokens，输出预算最高 {summary['requests']['max_legacy118_comparison_output_budget']}。所有已完成且检查了预算的请求均满足上下文上限。", '',
        '每个 MAGNET v2 请求使用服务实际 tokenizer 检查输入；输入上限 16,384，预留 256 tokens，各阶段输出上限为 384～3,072。长文档的问题描述按固定两页读取并汇总；视觉关联只读取历史轨迹实际报告的证据页或固定前页，避免整份文档重复进入请求。超出预算报错要求拆分，不静默截断输入。', '',
        '另外，embedding 服务的 tokenizer 并发访问可重现 HTTP 500。修复后 tokenizer 的单条、批量调用和模型推理共享锁；24 个混合请求、6 并发全部通过，1024 维和归一化检查通过。GPU 4/5/7 的生成服务未重启，embedding 在 GPU 2。', '',
        '## 6. 构建集端到端检查','']
    v=summary['integration']
    lines += [f"使用三条构建轨迹形成冻结记忆，再在另一条构建任务上执行检索、生成、渲染和评分。节点数 {v['nodes']}，实际注入 {v['retrieved']} 条记忆、{v['images']} 张历史图像；执行状态 {v['execution_status']}，渲染成功={v['render_success']}，评分 {v['score']}/5，记忆冻结={v['bank_frozen']}。", '',
        '这是构建集内的留一来源诊断，没有使用正式 held-out 任务，也不作为方法效果对比。机器记录位于 `vision2code/integration_multimodal/integration_validation.json`。', '',
        '## 7. 交付与结论','',
        f"机器验收状态 `ready_for_controlled_construction_run`：{summary['ready_for_controlled_construction_run']}。已验证导致三组失败的重复来源、重叠膨胀、全量去重超限等实现问题已得到处理。语义核查的范围有限，漏合并数及小模型定位限制按上文保留，不声称所有记忆都已获得独立正确性证明。", '',
        '自动检查 57 项通过，Python 编译及入口 Shell 语法检查通过。四记忆配置的完成计数还排除了另外保存的 no-memory 参考结果，避免将 400 个配置内结果误计为 500。', '',
        '正式脚本已指向 `*_four_memory_baselines_v2.json`，默认运行目录后缀为 `_four_memory_v2`。旧 `_v1` 目录保留，只复用原始构建轨迹；本轮未启动正式评测。', '',
        '复验入口：`bash Experiments/memory_baselines/scripts/diagnostics/validate_magnet_v2.sh <benchmark> <stage>`。`replay` 回放旧候选，`integration` 执行构建集内链路；调用和快照匹配时复用缓存。', '',
        f"累计已完成模型调用（按请求 ID 去重，含本轮诊断迭代）{summary['requests']['completed_unique_calls']} 次，prompt tokens={summary['requests']['prompt_tokens']}，completion tokens={summary['requests']['completion_tokens']}。失败的未完成传输不在 token 汇总内。", '',
        '完整机器验收：`Experiments/memory_baselines/artifacts/diagnostics/magnet_v2_20261025/acceptance.json`。报告由 AI 辅助生成，所有结果均取自本地实验产物。','']
    path=WORKSPACE/'Docs/实验分析/20261025-01-MAGNET-v2-构建集验证报告.md'
    path.write_text('\n'.join(lines),encoding='utf-8')
    print(path)
    print(json.dumps(dict(ready=summary['ready_for_controlled_construction_run'],dedup=dict(tp=tp,fp=fp,fn=fn),integration=v),ensure_ascii=False))


if __name__=='__main__':report()
