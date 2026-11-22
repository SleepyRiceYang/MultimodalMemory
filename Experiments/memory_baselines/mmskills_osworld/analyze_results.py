"""Offline audit of the four completed MMSkills arms; never launches experiments."""
import collections
import csv
import hashlib
import itertools
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import run as protocol

ROOT = protocol.ROOT
ART = ROOT / 'Experiments/memory_baselines/artifacts/mmskills_osworld'
OUT = ART / 'analysis_20261111_v1'
REPORT = ROOT / 'Docs/实验分析/20261111-MMSkills四组正式实验结果分析.md'
RUNS = {'no_skill': ART / 'formal_40_three_20261110_v1',
        'text_skill': ART / 'formal_40_three_20261110_v1',
        'visual_skill': ART / 'formal_40_three_20261110_v1',
        'visual_skill_retrieval': ART / 'formal_40_dense_20261111_v1'}
LABEL = {'no_skill': 'No-Skill', 'text_skill': 'Text-Skill', 'visual_skill': 'Visual-Skill（映射）',
         'visual_skill_retrieval': 'Visual-Skill（检索）'}


def read(path):
    return json.loads(Path(path).read_text())


def avg(xs):
    return sum(xs) / len(xs) if xs else None


def percent(x):
    return '—' if x is None else str((Decimal(str(x))*100).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))+'%'


def exact_mcnemar(gains, losses):
    n = gains + losses
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(gains, losses) + 1)) / 2**n) if n else 1.


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    result, previous = [None] * len(ps), 0.
    for rank, i in enumerate(order):
        previous = min(1., max(previous, ps[i] * (len(ps) - rank)))
        result[i] = previous
    return result


def analyze():
    split = read(protocol.SPLITS / 'test_40.json')
    expected = {(d, t) for d, ids in split.items() for t in ids}
    assert len(expected) == 40
    configs = {c: read(p / 'config.json') for c, p in RUNS.items()}
    for config in configs.values():
        assert config['tasks'] == split
    for k in ('model', 'max_steps', 'max_tokens', 'context_window', 'action_wait', 'skills_dir', 'vm_path'):
        values = [c[k] for c in configs.values()]
        assert all(v == values[0] for v in values), k
    assert configs['visual_skill']['prompt_source_sha256'] == configs['visual_skill_retrieval']['prompt_source_sha256']
    data, manifest, errors, audit_warnings = {}, {}, [], []
    for condition, root in RUNS.items():
        records = {}
        files = list((root / condition).glob('*/*/result.json'))
        assert {(p.parent.parent.name, p.parent.name) for p in files} == expected
        for path in files:
            row = read(path)
            key, folder = (row['domain'], row['task']), path.parent
            manifest[str(path.relative_to(ROOT))] = protocol.digest(path)
            calls = read(folder / 'api/calls.json') if (folder / 'api/calls.json').exists() else []
            usage = read(folder / 'skill_usage_summary.json') if (folder / 'skill_usage_summary.json').exists() else {}
            selection = read(folder / 'selection.json') if (folder / 'selection.json').exists() else None
            evaluation = read(folder / 'evaluation.json') if (folder / 'evaluation.json').exists() else None
            if row['status'] == 'completed':
                assert evaluation and evaluation['status'] == 'valid'
                assert row['score'] == evaluation['score']
                assert float((folder / 'result.txt').read_text()) == row['score']
            if evaluation:
                for g in evaluation['getters']:
                    for f in g.get('files', []):
                        artifact = Path(f['path'])
                        if not artifact.exists():
                            audit_warnings.append({'file': f['path'], 'issue': 'missing_saved_evidence'})
                        elif protocol.digest(artifact) != f['sha256']:
                            audit_warnings.append({'file': f['path'], 'issue': 'evidence_hash_mismatch'})
            if row['status'] != 'completed':
                error = {k: row.get(k) for k in ('condition', 'domain', 'task', 'status', 'error', 'evaluation_issues')}
                error['artifact_dir'] = str(folder)
                errors.append(error)
            records[key] = {'row': row, 'folder': folder, 'calls': calls, 'usage': usage,
                            'selection': selection, 'evaluation': evaluation}
        data[condition] = records
        saved = read(root / 'summary.json')
        saved_rows = {(r['condition'], r['domain'], r['task']): r for r in saved['rows']}
        for (d, t), record in records.items():
            assert saved_rows[(condition, d, t)] == record['row']
    valid = {c: {k for k, v in records.items() if v['row']['status'] == 'completed'} for c, records in data.items()}
    common = set.intersection(*valid.values())
    smoke = {(d, t) for d, ids in read(protocol.SPLITS / 'smoke_3.json').items() for t in ids}
    overview, domain_scores, flat = {}, {}, []
    for c, records in data.items():
        values = [records[k]['row']['score'] for k in valid[c]]
        common_values = [records[k]['row']['score'] for k in common]
        unseen_values = [records[k]['row']['score'] for k in common - smoke]
        calls = [call for record in records.values() for call in record['calls']]
        total_score = sum(values)
        overview[c] = {'planned': 40, 'valid': len(values), 'invalid': 40-len(values),
            'status_counts': dict(collections.Counter(r['row']['status'] for r in records.values())),
            'full_successes': sum(v == 1 for v in values), 'partial_scores': [v for v in values if 0 < v < 1],
            'sum_valid_scores': total_score, 'mean_valid': avg(values),
            'fixed40_lower_bound': total_score / 40, 'fixed40_upper_bound': (total_score+40-len(values))/40,
            'common_valid_n': len(common), 'common_mean': avg(common_values),
            'common_full_successes': sum(v == 1 for v in common_values),
            'common_excluding_smoke_n': len(unseen_values), 'common_excluding_smoke_mean': avg(unseen_values),
            'api_calls': len(calls), 'api_errors': sum(call['status'] != 'completed' for call in calls),
            'recorded_prompt_tokens': sum((call.get('usage') or {}).get('prompt_tokens', 0) for call in calls),
            'recorded_completion_tokens': sum((call.get('usage') or {}).get('completion_tokens', 0) for call in calls),
            'consulted_tasks': sum(r['usage'].get('skill_branch_invocations',0)>0 for r in records.values()),
            'consultation_calls': sum(r['usage'].get('skill_branch_invocations',0) for r in records.values()),
            'reference_images_loaded': sum(r['usage'].get('total_loaded_images',0) for r in records.values()),
            'steps_mean_valid': avg([records[k]['row']['steps'] for k in valid[c]]),
            'seconds_mean_valid': avg([records[k]['row']['seconds'] for k in valid[c]])}
        domain_scores[c] = {}
        for domain, ids in split.items():
            ds = [records[(domain,t)]['row']['score'] for t in ids if (domain,t) in valid[c]]
            domain_scores[c][domain] = {'planned':len(ids),'valid':len(ds),'mean_valid':avg(ds),
                                      'full_successes':sum(v==1 for v in ds)}
        for key in sorted(expected):
            r = records[key]
            flat.append({'condition':c, 'domain':key[0], 'task':key[1], 'status':r['row']['status'],
                'score':r['row'].get('score'), 'raw_score':r['row'].get('raw_score'),
                'common_valid':key in common, 'smoke_development_task':key in smoke,
                'selected_skills':json.dumps((r['selection'] or {}).get('selected_skills',[])),
                'skill_consults':r['usage'].get('skill_branch_invocations',0),
                'reference_images':r['usage'].get('total_loaded_images',0),
                'steps':r['row'].get('steps'), 'model_calls_recorded':len(r['calls']),
                'api_errors':sum(call['status']!='completed' for call in r['calls'])})
    pairs = []
    for a,b in itertools.combinations(RUNS,2):
        changes = [(data[b][k]['row']['score'],data[a][k]['row']['score']) for k in common]
        gains = sum(x==1 and y!=1 for x,y in changes)
        losses = sum(x!=1 and y==1 for x,y in changes)
        pairs.append({'a':a,'b':b,'n':len(common),'b_gains_full_success':gains,'b_losses_full_success':losses,
                      'b_minus_a_mean_score':avg([x-y for x,y in changes]),'p_exact_mcnemar':exact_mcnemar(gains,losses)})
    for pair,p in zip(pairs,holm([p['p_exact_mcnemar'] for p in pairs])):
        pair['p_holm_6']=p
    retrieval_rows=[]
    for key in sorted(expected):
        mapped=data['visual_skill'][key]['selection']['selected_skills']
        assert mapped==data['text_skill'][key]['selection']['selected_skills']
        d=data['visual_skill_retrieval'][key]
        sel=d['selection']; a=set(mapped);b=set(sel['selected_skills']);top={r['skill'] for r in sel['candidates']}
        assert not sel['mapping_used'] and len(b)<=3 and len(sel['candidates'])==10
        retrieval_rows.append({'domain':key[0],'task':key[1],'mapped':mapped,'selected':sel['selected_skills'],
            'exact_set_match':a==b,'any_match':bool(a&b),'top10_any_match':bool(a&top),'top10_all_match':a<=top,
            'map_recall':len(a&b)/len(a),'map_precision':len(a&b)/len(b) if b else None,
            'mapping_size':len(a),'selected_size':len(b),'intersection':len(a&b),
            'consulted':d['usage'].get('skill_branch_invocations',0)>0,
            'score':d['row'].get('score'),'status':d['row']['status']})
    retrieval={'size_counts':dict(collections.Counter(x['selected_size'] for x in retrieval_rows)),
        'exact_sets':sum(x['exact_set_match'] for x in retrieval_rows),'any_match':sum(x['any_match'] for x in retrieval_rows),
        'top10_any':sum(x['top10_any_match'] for x in retrieval_rows),'top10_all':sum(x['top10_all_match'] for x in retrieval_rows),
        'macro_map_recall':avg([x['map_recall'] for x in retrieval_rows]),
        'micro_map_recall':sum(x['intersection'] for x in retrieval_rows)/sum(x['mapping_size'] for x in retrieval_rows),
        'macro_map_precision_nonempty':avg([x['map_precision'] for x in retrieval_rows if x['map_precision'] is not None]),
        'rows':retrieval_rows}
    vd=[]
    for key in sorted(valid['visual_skill']&valid['visual_skill_retrieval']):
        a,b=(data[c][key]['row']['score'] for c in ('visual_skill','visual_skill_retrieval'))
        if a!=b:
            conf=read(protocol.OSWORLD/'evaluation_examples/examples'/key[0]/(key[1]+'.json'))
            vd.append({'domain':key[0],'task':key[1],'instruction':conf['instruction'],
                       'mapped_score':a,'retrieved_score':b,
                       'mapped_skills':data['visual_skill'][key]['selection']['selected_skills'],
                       'retrieved_skills':data['visual_skill_retrieval'][key]['selection']['selected_skills']})
    result={'verification_status':'ANALYZED; no experiment reruns','overview':overview,'domains':domain_scores,
        'common_valid_tasks':[list(k) for k in sorted(common)],'common_excluding_smoke_n':len(common-smoke),
        'paired_tests_common29':pairs,'retrieval':retrieval,'visual_vs_retrieval_changed_tasks':vd,
        'errors':errors,'evidence_warnings':audit_warnings,'source_result_sha256':manifest}
    OUT.mkdir(parents=True,exist_ok=True)
    protocol.write(OUT/'statistics.json',result)
    with (OUT/'per_task.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    render(result)
    print(json.dumps({'overview':overview,'paired_tests':pairs,'retrieval':{k:v for k,v in retrieval.items() if k!='rows'},
                      'evidence_warnings':audit_warnings},ensure_ascii=False,indent=2))


def render(s):
    o=s['overview'];lines=['# MMSkills 四组正式实验结果分析','', '## Material Passport','',
        '- Verification Status：ANALYZED。仅对既有文件做离线统计与证据核对，没有重跑任务。',
        '- 范围：固定 40 题、4 组、160 条任务记录；不混入 smoke 或分支诊断记录。',
        '- 数据：前三组 `formal_40_three_20261110_v1`；第四组 `formal_40_dense_20261111_v1`。',
        '- 脚本：[analyze_results.py](../../Experiments/memory_baselines/mmskills_osworld/analyze_results.py)。',
        '- 明细：[statistics.json](../../Experiments/memory_baselines/artifacts/mmskills_osworld/analysis_20261111_v1/statistics.json)、[per_task.csv](../../Experiments/memory_baselines/artifacts/mmskills_osworld/analysis_20261111_v1/per_task.csv)。','',
        '## 1. 主要结果与计分口径','',
        '**四组已结束，但有 22/160 条记录不是有效评分。Visual-Skill 映射组在本次样本上最高，检索组次之；这不是完整 OSWorld 的复现分数。**',
        '官方得分含连续分数：两个视觉条件在同一图像任务上分别得到 0.777937 和 0.791932；均分保留原值，完全成功另按 score=1 计数。','',
        '| 条件 | 有效/40 | 异常 | 完全成功 | 有效子集均分 | 固定40题均分界限 |',
        '|---|---:|---:|---:|---:|---:|']
    for c in RUNS:
        x=o[c];lines.append(f"| {LABEL[c]} | {x['valid']}/40 | {x['invalid']} | {x['full_successes']} | {percent(x['mean_valid'])} | [{percent(x['fixed40_lower_bound'])}, {percent(x['fixed40_upper_bound'])}] |")
    lines += ['', '界限仅把未知评分置于 [0,1] 计算：下界是假设所有异常为0，上界是假设全部为1，并不是把异常重新判成有效分数，也不是置信区间。有效子集分母不同，不能仅凭该均分比较。所有原始 summary 的全量均分均为 null，退出码为1。','',
        '## 2. 相同任务上的配对比较','',
        f"四组共同有效 **{len(s['common_valid_tasks'])} 题**，均分如下。共同有效任务是事后条件筛选，不是随机独立缺失；应与异常分布同时看。",'',
        '| 条件 | 均分 | 完全成功 | 排除3题smoke开发任务后的共同有效均分 |','|---|---:|---:|---:|']
    for c in RUNS:
        x=o[c];lines.append(f"| {LABEL[c]} | {percent(x['common_mean'])} | {x['common_full_successes']}/{x['common_valid_n']} | {percent(x['common_excluding_smoke_mean'])}（n={x['common_excluding_smoke_n']}） |")
    lines += ['', '探索性双侧 exact McNemar 检验比较“完全成功”二元结果，单位为相同任务；6 个比较用 Holm 校正。它不检验连续原始均分，不把部分得分算成完全成功。','',
        '| A → B | B新增成功/丢失成功 | 均分差(pp) | 原始p | Holm校正p |','|---|---:|---:|---:|---:|']
    for p in s['paired_tests_common29']:
        lines.append(f"| {LABEL[p['a']]} → {LABEL[p['b']]} | {p['b_gains_full_success']}/{p['b_losses_full_success']} | {100*p['b_minus_a_mean_score']:+.2f} | {p['p_exact_mcnemar']:.6f} | {p['p_holm_6']:.6f} |")
    lines += ['', '六项校正后仅映射视觉组相对No-Skill的完全成功差异落在0.05以内（p=0.046875），接近阈值；仍受事后有效子集选择、单次运行与小样本限制，作为探索性结果，不作为独立确认性结论。', '',
        '两种视觉条件的有效题集完全相同，为37题，可直接比较：映射组37.24%，检索组31.87%，检索组低5.37个百分点；完全成功少2题。原始分数有5题不同：检索组1题由0变1、3题由1变0、1题部分得分略高。其余32题相同。该单次小样本差异不能当作检索策略的稳定劣势。','',
        '## 3. 分域得分','', '单元格为“有效子集均分（有效数/计划数）”。每域仅2–10题，且分母可能不同；不做分域显著性检验。','',
        '| 域 | No-Skill | Text-Skill | Visual映射 | Visual检索 |','|---|---:|---:|---:|---:|']
    for d in s['domains']['no_skill']:
        cells=[]
        for c in RUNS:
            x=s['domains'][c][d];cells.append(f"{percent(x['mean_valid'])} ({x['valid']}/{x['planned']})")
        lines.append('| '+d+' | '+' | '.join(cells)+' |')
    lines += ['', '映射视觉组的有效完全成功分布跨多个域，而非仅由一个域贡献；Calc 中 Text-Skill 为2/5成功、映射视觉组为0/5，说明总体排名不意味着逐域或逐题都更好。','',
        '## 4. 记忆利用与调用成本','',
        '| 条件 | 咨询任务/40 | 分支调用 | 加载参考图次数 | 记录的VLM请求 | 失败请求 |','|---|---:|---:|---:|---:|---:|']
    for c in RUNS:
        x=o[c];lines.append(f"| {LABEL[c]} | {x['consulted_tasks']}/40 | {x['consultation_calls']} | {x['reference_images_loaded']} | {x['api_calls']} | {x['api_errors']} |")
    lines += ['', '统计覆盖全部尝试，包括异常任务；图片计数是加载次数，不保证为唯一图像数。第四组VLM请求额外包含40次任务前技能选择，Embedding不在VLM token统计内。',
        '**Visual-Skill 高分不能直接解释为加载图片带来的提升**：映射视觉组仅11题咨询技能，共加载5张参考图；与文本组同时存在主提示词、状态提示、调用策略和规划结构差异。它们是完整运行机制对照，不是只切换图片的严格消融。',
        '咨询与否由Agent根据状态决定，不能用“咨询任务成功率”和“不咨询任务成功率”的相关性推断技能造成的效果。耗时也不能直接比较：前三组与第四组使用不同后端端口/并行拓扑，且部分时间共用VLM、Embedding与VLM可能共卡。','',
        '## 5. 第四组召回与作者映射的一致性','']
    q=s['retrieval']
    lines += [f"- 选择数量：{q['size_counts']}，其中6题为空；仅34题选到至少一个技能。",
        f"- 与映射集合完全一致：{q['exact_sets']}/40；至少一个交集：{q['any_match']}/40。",
        f"- MiniCPM Top-10 至少覆盖一个映射技能：{q['top10_any']}/40；覆盖全部映射技能：{q['top10_all']}/40。",
        f"- 最终选择相对映射的宏平均 recall：{percent(q['macro_map_recall'])}；micro recall：{percent(q['micro_map_recall'])}。",
        f"- 非空选择的宏平均 precision：{percent(q['macro_map_precision_nonempty'])}（空集合另行报告，不悄悄视为满分）。",'',
        '这些是对作者映射的一致性指标，不是真实相关性金标准。映射以外的技能不必然无用；映射本身也可能不完备。32题在Top-10有交集，最终27题有交集，说明候选阶段至少8题存在覆盖缺口，后续LLM选择又减少了映射重合覆盖；不能仅凭这一点给出因果归因。','',
        '### 两种视觉条件的差异任务','', '| 域/任务ID | 映射得分 | 检索得分 |','|---|---:|---:|']
    for x in s['visual_vs_retrieval_changed_tasks']:
        lines.append(f"| {x['domain']}/{x['task']} | {x['mapped_score']:.6f} | {x['retrieved_score']:.6f} |")
    lines += ['', '完整任务指令、候选列表及差异记录见 statistics.json。进一步查看实际选择和咨询：',
        '- Calc变正确任务：两组候选完全相同；映射组咨询1次仍失败，检索组没有咨询而成功。因此该收益不能归因于召回到更好的技能。',
        '- Impress变错误任务：检索组保留幻灯片编号技能，但丢掉另一项文本格式技能；两组都未咨询，无法仅凭得分证明候选缺失导致失败。',
        '- Thunderbird变错误任务：映射是创建邮件过滤规则，检索变成对子文件夹应用过滤器；两组均未咨询。候选偏向不同功能，但因果关系未隔离。',
        '- VS Code变错误任务：检索误选Writer制表位/段落对齐技能，映射是VS Code块编辑。该任务指令只写增加行缩进，没有应用名称，暴露纯任务文本召回的应用消歧不足；额外加入应用状态信息需另设对照，不能直接声称仍是同一检索设定。',
        '- Multi-app部分得分略增：检索额外加入VLC截帧与Impress布局技能，两组都未咨询，不能把微小分差认定为这些技能带来的作用。',
        '相同候选也可能产生不同轨迹，单次运行不能排除推理调度和状态分歧。','',
        '## 6. 异常分布与证据可靠性','',
        '- No-Skill：5条评分异常、3条执行异常；Text-Skill：6条评分异常、2条执行异常；两种视觉组各3条评分异常。',
        '- 四组共同的3道Multi-app任务：compressed.jpeg、notes.docx 或 contacts.xlsx 等目标文件不可获取。文件未生成可能就是任务执行失败，不能一律称为外部基础设施故障；本报告保留运行器的invalid分类，并给出未知分数界限。',
        '- Text-Skill另有Chrome评估的tldextract缓存锁超时；有页面信息读取失败；No-Skill与Text-Skill的GIMP主题任务仍出现配置输入缺失。',
        '- 执行异常包括Agent未返回动作，以及API超时；记录到No-Skill 3次、Text-Skill 6次失败VLM请求。第四组无API失败。',
        f"- 逐题result与原summary一致，所有有效result.txt与evaluation.json分数一致。评分文件存在性/哈希警告：{len(s['evidence_warnings'])} 条；具体见 statistics.json。",'',
        '## 7. 可支持的结论与后续边界','',
        '本次40题小样本中，完整Visual-Skill机制的得分高于文本和无技能条件；共同有效任务上保持这一排序。任务文本检索组接近但低于固定作者映射的视觉组，且存在候选覆盖损失。尚不能据此断言“视觉图像是提升原因”、检索策略稳定优劣或推广至全360题。',
        '需要优先处理/确认22条异常的性质，特别是任务输出缺失与基础设施失败的区别。若补跑，应预先固定规则并保存历史失败，不因看到得分而挑选替代轨迹；本轮没有自动补跑。',
        '样本只有一次运行、每域题量小，3道smoke题已用于开发；附带排除这些题的共同有效敏感性结果，但仍是探索性分析。未用独立开发集固定所有检索参数，不将结果包装为确认性统计。','',
        '## 8. 统计解释检查（11/11）','',
        '| 检查 | 本次处理 |','|---|---|',
        '| Simpson反转 | 给出分域结果；域内方向不一致，不声称所有域同向 |',
        '| 生态谬误 | 推断单位限于任务，不从域均分推断每题获益 |',
        '| Berkson选择偏差 | 共同有效29题为筛选子集，披露选择限制 |',
        '| Collider偏差 | 不把是否咨询作为随机对照或回归控制来推断因果 |',
        '| 基率忽略 | 同时报题数、完全成功数、分数与异常 |',
        '| 均值回归 | 非极端任务重测改进设计；单次结果不做演化解释 |',
        '| 幸存者偏差 | 披露全部22条异常，报告40题界限与共同有效结果 |',
        '| 多重搜索 | 6项配对检验统一Holm校正，不挑显著域 |',
        '| 分析路径自由度 | 明示探索性与多种分母，不据结果改变有效性口径 |',
        '| 相关与因果 | 不归因于单独图像/检索模块，报告机制和后端混杂 |',
        '| 反向因果 | 调用可能由困难状态触发，调用与得分关系不作方向性解释 |', '']
    REPORT.write_text('\n'.join(lines))


if __name__=='__main__':
    analyze()
