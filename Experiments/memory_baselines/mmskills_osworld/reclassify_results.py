"""Apply outcome-aware scoring annotations without modifying original experiment artifacts."""
import collections
import csv
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import analyze_results as previous

ROOT, ART = previous.ROOT, previous.ART
OUT = ART / 'analysis_20261111_v2'
REPORT = ROOT / 'Docs/实验分析/20261111-MMSkills四组结果重新分类与统计.md'


def classify(folder):
    raw = previous.read(folder / 'result.json')
    calls = previous.read(folder / 'api/calls.json') if (folder / 'api/calls.json').exists() else []
    api_errors = [c for c in calls if c['status'] != 'completed']
    evaluation = previous.read(folder / 'evaluation.json') if (folder / 'evaluation.json').exists() else None
    log = (folder / 'runtime.log').read_text()
    score, reason, evidence = None, None, []
    if raw['status'] == 'completed':
        score, reason = raw['score'], 'original_valid_evaluation'
    elif evaluation and any(i['type'] == 'evaluation_exception' for i in evaluation['issues']):
        assert 'tldextract' in str(evaluation['issues']) and 'Timeout' in str(evaluation['issues'])
        reason, evidence = 'environment_evaluator_cache_lock_timeout', evaluation['issues']
    elif evaluation and any(g['getter'] == 'get_active_tab_info' and g.get('returned_none') for g in evaluation['getters']):
        assert 'TargetClosedError' in log and 'socket hang up' in log
        reason = 'environment_browser_evaluation_connection_failure'
        evidence = ['TargetClosedError', 'BrowserType.connect_over_cdp: socket hang up']
    elif evaluation and any(g['getter'] == 'get_background_image_in_slide' and g.get('returned_none') for g in evaluation['getters']):
        # The slide file exists; absence of the requested background is the task outcome.
        files = list((folder / 'env_cache').rglob('Robotic_Workshop_Infographics.pptx'))
        assert len(files) == 1
        with zipfile.ZipFile(files[0]) as z:
            slide = ET.fromstring(z.read('ppt/slides/slide2.xml'))
            bg = list(slide.iter('{http://schemas.openxmlformats.org/presentationml/2006/main}bgPr'))
            assert not bg
        score, reason = 0., 'goal_background_not_created'
        evidence = [str(files[0]), 'Readable slide2.xml; background property count=0']
    elif evaluation and any(g['getter'] in ('get_vm_file', 'get_gimp_config_file') and g.get('returned_none')
                            for g in evaluation['getters']):
        # Confirm HTTP 404, not a transport timeout, and retain the actual missing paths.
        assert 'Status code: 404' in log
        missing = [t['guest_path'] for t in evaluation['file_transfers'] if t.get('returned_none')]
        assert missing and evaluation['raw_score'] == 0
        score, reason = 0., 'goal_output_or_saved_configuration_missing'
        evidence = missing
    elif raw.get('error') == 'Agent returned no actions':
        trajectory = folder / 'traj.jsonl'
        assert not trajectory.exists() or not trajectory.read_text().strip()
        if api_errors:
            assert all('timed out' in c.get('error','') for c in api_errors)
            reason = 'environment_model_timeout_before_any_action'
            evidence = [c.get('error') for c in api_errors]
        else:
            assert len(calls) > 0 and all(c['status'] == 'completed' for c in calls)
            score, reason = 0., 'goal_agent_failed_to_produce_executable_action'
            evidence = ['API responses completed, but no GUI actions executed; output/parser failure']
    else:
        raise ValueError(f'Unreviewed error classification: {folder}')
    outcome = ('environment_error_unscorable' if score is None else
               'goal_completed' if score == 1 else 'partial_goal_completion' if score > 0 else 'goal_not_completed')
    return {'condition':raw['condition'], 'domain':raw['domain'], 'task':raw['task'],
            'original_status':raw['status'], 'original_score':raw.get('score'), 'raw_score':raw.get('raw_score'),
            'outcome':outcome, 'score':score, 'reason':reason, 'evidence':evidence,
            'api_error_count':len(api_errors), 'artifact_dir':str(folder),
            'source_result_sha256':previous.protocol.digest(folder/'result.json')}


def main():
    records = {}
    for condition, root in previous.RUNS.items():
        files = list((root/condition).glob('*/*/result.json'))
        assert len(files) == 40
        records[condition] = {(p.parent.parent.name,p.parent.name):classify(p.parent) for p in files}
    keys = [set(v) for v in records.values()]
    assert all(k==keys[0] for k in keys)
    valid = {c:{k for k,v in rows.items() if v['score'] is not None} for c,rows in records.items()}
    common = set.intersection(*valid.values())
    summary, domains = {}, {}
    for c, rows in records.items():
        scores = [v['score'] for v in rows.values() if v['score'] is not None]
        values = [rows[k]['score'] for k in common]
        summary[c] = {'planned':40, 'scorable':len(scores), 'environment_errors':40-len(scores),
            'outcomes':dict(collections.Counter(v['outcome'] for v in rows.values())),
            'mean_scorable':previous.avg(scores), 'sum_scores':sum(scores),
            'fixed40_lower_bound':sum(scores)/40, 'fixed40_upper_bound':(sum(scores)+40-len(scores))/40,
            'common_n':len(common), 'common_mean':previous.avg(values),
            'common_full_successes':sum(v==1 for v in values)}
        domains[c] = {}
        for domain in sorted({d for d,t in rows}):
            dr = [v for (d,t),v in rows.items() if d==domain]
            ds = [v['score'] for v in dr if v['score'] is not None]
            domains[c][domain] = {'planned':len(dr),'scorable':len(ds),'mean':previous.avg(ds)}
    pairs = []
    for a,b in itertools.combinations(previous.RUNS,2):
        changes=[(records[b][k]['score'],records[a][k]['score']) for k in common]
        gains=sum(x==1 and y!=1 for x,y in changes);losses=sum(x!=1 and y==1 for x,y in changes)
        pairs.append({'a':a,'b':b,'n':len(common),'b_gains':gains,'b_losses':losses,
                      'b_minus_a_mean':previous.avg([x-y for x,y in changes]),
                      'p_exact_mcnemar':previous.exact_mcnemar(gains,losses)})
    for pair,p in zip(pairs,previous.holm([p['p_exact_mcnemar'] for p in pairs])):
        pair['p_holm_6']=p
    rows=[v for c in records for k,v in sorted(records[c].items())]
    relabeled=[r for r in rows if r['original_status']!='completed' and r['score'] is not None]
    env=[r for r in rows if r['score'] is None]
    assert len(rows)==160 and len(relabeled)==17 and len(env)==5
    stats={'policy':'Only environment-caused inability to assess is exceptional; unmet goal is score 0',
        'verification_status':'ANALYZED, no experiment reruns', 'scorable':155,'environment_errors':5,
        'reclassified_goal_failures':17,'summary':summary,'common_n':len(common),
        'common_tasks':[list(k) for k in sorted(common)],'paired_tests':pairs,'domains':domains,
        'rows':rows,'reclassified':relabeled,'environment_error_records':env}
    OUT.mkdir(parents=True,exist_ok=True)
    previous.protocol.write(OUT/'statistics.json',stats)
    with (OUT/'per_task.csv').open('w',newline='') as f:
        fields=list(rows[0]);writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    render(stats)
    print(json.dumps({'summary':summary,'common_n':len(common),'paired_tests':pairs},ensure_ascii=False,indent=2))


def render(s):
    pct=previous.percent;label=previous.LABEL
    lines=['# MMSkills 四组结果：按“未完成目标”口径重新分类','',
        '## Material Passport','',
        '- Verification Status：ANALYZED。对既有日志、API响应、评分记录和PPT文件做离线复核，没有重跑实验。',
        '- 本报告取代上一版把所有getter None排除的统计口径；原始实验文件、原summary、上一版统计均保留。',
        '- 新统计：[statistics.json](../../Experiments/memory_baselines/artifacts/mmskills_osworld/analysis_20261111_v2/statistics.json)、[逐题标注CSV](../../Experiments/memory_baselines/artifacts/mmskills_osworld/analysis_20261111_v2/per_task.csv)。','',
        '## 1. 分类规则','',
        '只有环境/服务/评估器故障导致无法确定任务结果时标异常。任务未生成要求的文件、未设置目标属性、未保存配置，或Agent在API正常时没有产生可执行动作，标注为 **未完成目标，0分**。部分分数原样保留。',
        'API错误事件与可否评分分开：若最终产物已取回且能确认目标未完成，仍计0分，并另记API错误。未执行官方metric但因Agent输出格式失败而无任何动作的记录，本次按执行失败规则赋0，明确区别于官方原始返回值。','',
        '## 2. 改判明细','',
        '- 原22条非completed记录：17条改为未完成目标，5条保留环境异常；总计155/160条可评分。',
        '- 12条缺失目标文件：四组各3题，要求输出compressed.jpeg、notes.docx和联系人导出文件。VM读取端点返回404，目标路径不存在，计任务失败。',
        '- 2条GIMP配置未落盘：No-Skill和Text-Skill均未取得gimprc，原官方metric返回0；按未完成保存目标计0，不再单独排除。',
        '- 2条PPT无背景图：PPT成功读取且第二页XML没有背景属性，任务目标明确未完成，计0；其中Text-Skill虽有6次API超时，最终产物仍可判定，保留0分。',
        '- 1条Text-Skill邮件任务：10次API响应均正常，但Agent在咨询技能后重复返回不合规范的单反引号动作，解析未产生任何GUI动作，计Agent执行失败0分。','',
        '保留的5条环境异常：No-Skill三题第一次模型请求均300秒超时且无动作；Text-Skill一题Chrome评估发生TargetClosedError/CDP socket hang up，一题tldextract缓存锁超时。','',
        '## 3. 更新后的总体结果','',
        '| 设置 | 可评分/40 | 未完成目标（0分） | 完全成功 | 部分得分 | 环境异常 | 可评分均分 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for c,x in s['summary'].items():
        o=x['outcomes'];lines.append(f"| {label[c]} | {x['scorable']}/40 | {o.get('goal_not_completed',0)} | {o.get('goal_completed',0)} | {o.get('partial_goal_completion',0)} | {x['environment_errors']} | {pct(x['mean_scorable'])} |")
    lines += ['', '**前三组与第四组的完全成功数未改变，改变的是此前被不当排除的失败任务重新进入分母。** 两个视觉组全部40题现在均可评分，完整40题均分分别为34.44%和29.48%；No-Skill和Text-Skill仍有环境异常，不能把其可评分子集均分当作完整40题分数。','',
        '| 设置 | 40题均分下界 | 40题均分上界 |','|---|---:|---:|']
    for c,x in s['summary'].items():
        lines.append(f"| {label[c]} | {pct(x['fixed40_lower_bound'])} | {pct(x['fixed40_upper_bound'])} |")
    lines += ['', '上下界只对无法评分的环境异常取[0,1]；不是置信区间，也未擅自把环境异常计0。','',
        f"## 4. 共同可评分的{s['common_n']}题",'',
        '| 设置 | 共同题集均分 | 完全成功 |','|---|---:|---:|']
    for c,x in s['summary'].items():
        lines.append(f"| {label[c]} | {pct(x['common_mean'])} | {x['common_full_successes']}/{s['common_n']} |")
    lines += ['', '探索性exact McNemar比较完全成功二元结果，6项Holm校正；不检验含部分分数的连续均分。','',
        '| A → B | B新增/丢失完全成功 | 均分差(pp) | 原始p | Holm p |','|---|---:|---:|---:|---:|']
    for p in s['paired_tests']:
        lines.append(f"| {label[p['a']]} → {label[p['b']]} | {p['b_gains']}/{p['b_losses']} | {p['b_minus_a_mean']*100:+.2f} | {p['p_exact_mcnemar']:.6f} | {p['p_holm_6']:.6f} |")
    lines += ['', '单次小样本、后台资源与分支机制差异仍存在；不将总体差异归因于单独图像或检索模块，不把共同可评分子集当作完整benchmark随机样本。原报告的召回一致性、技能调用次数和分域异质性结论未因失败重分类改变；原有11项统计解释检查仍适用。','',
        '## 5. 状态与后续','',
        '只生成新的分析标注，没有修改原始result.json、evaluation.json或result.txt，也没有补跑任务。重新统计脚本为 [reclassify_results.py](../../Experiments/memory_baselines/mmskills_osworld/reclassify_results.py)。',
        '后续运行时应按getter/metric的语义区分“目标不存在”和“环境无法读取”，不再统一把None判为异常。当前旧运行器的历史标签没有被悄悄改写；新报告和逐题标注是本次解释口径的依据。','']
    REPORT.write_text('\n'.join(lines))


if __name__=='__main__':
    main()
