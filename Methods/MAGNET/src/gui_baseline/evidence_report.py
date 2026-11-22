"""Export auditable paired results, exact request prompts and screenshot trajectories."""
import argparse,csv,html,json,os
from pathlib import Path
from .common import read,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    run=Path(args.run).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    costs={c:{'calls':0,'prompt_tokens':0,'completion_tokens':0,'request_images':0} for c in ['B2','B3']}
    configs=[read(run/c/'config.json') for c in ['B2','B3'] if (run/c/'config.json').exists()]
    tasks=configs[0]['tasks'];rows=[];loaded={'B2':0,'B3':0};selected={'B2':0,'B3':0}
    for task in tasks:
        body=[f'<h1>{html.escape(task)}</h1>'];row={'task':task}
        for condition in ['B2','B3']:
            folder=run/condition/'tasks'/task;cp=folder/'checkpoint.json'
            if cp.exists():
                episode=read(cp)['episode'];row[condition+'_score']=episode['score'];row[condition+'_steps']=episode['steps']
                body.append(f'<h2>{condition}: score {episode["score"]}; steps {episode["steps"]}</h2>')
            else:row[condition+'_score']='pending';row[condition+'_steps']='';body.append(f'<h2>{condition}: incomplete</h2>')
            for step in sorted(folder.glob('step_[0-9][0-9][0-9]')):
                body.append(f'<section><h3>{condition} {step.name}</h3>')
                for image in ['before.png','after.png']:
                    f=step/image
                    if f.exists():body.append(f'<figure><img loading="lazy" src="{html.escape(os.path.relpath(f,out))}"><figcaption>{image}</figcaption></figure>')
                for role in ['planner','actor']:
                    evidence=step/(role+'_evidence.json')
                    if evidence.exists():
                        e=read(evidence);loaded[condition]+=len(e['loaded_images']);selected[condition]+=len(e['selected_images'])
                        if condition=='B2':assert not e['loaded_images'],'B2 history leakage'
                        body.append('<pre>'+html.escape(json.dumps(e,ensure_ascii=False,indent=2))+'</pre>')
                for role in ['observe','planner_evidence_selection','planner','actor_evidence_selection','actor','reflect']:
                    for filename in ['request.json','response.json']:
                        f=step/role/filename
                        if f.exists():
                            content=read(f)
                            if condition=='B2' and role in ['planner','actor'] and filename=='request.json':assert len(content['images'])==1
                            body.append(f'<details><summary>{role}/{filename}</summary><pre>{html.escape(json.dumps(content,ensure_ascii=False,indent=2))}</pre></details>')
                body.append('</section>')
        for condition in ['B2','B3']:
            folder=run/condition/'tasks'/task
            requests=list(folder.glob('step_*/*/request.json'))
            costs[condition]['calls']+=len(list(folder.glob('step_*/*/raw_response_*.json')))
            costs[condition]['request_images']+=sum(len(read(f)['images']) for f in requests)
            for f in folder.glob('step_*/*/raw_response_*.json'):
                usage=read(f)['usage']
                for field in ['prompt_tokens','completion_tokens']:costs[condition][field]+=usage[field]
        rows.append(row)
        (out/(task+'.html')).write_text('<!doctype html><meta charset="utf-8"><style>body{font:15px sans-serif;max-width:1400px;margin:auto}section{border-top:2px solid #789;padding:12px}figure{display:inline-block;width:44%}img{max-width:100%;max-height:650px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:12px}details{margin:8px}</style>'+''.join(body))
    with (out/'results.csv').open('w') as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    stats={c:{'completed':sum(r[c+'_score']!='pending' for r in rows),'successes':sum(r[c+'_score']==1 for r in rows),'selected_views':selected[c],'loaded_views':loaded[c]} for c in ['B2','B3']}
    write(out/'statistics.json',stats)
    write(out/'costs.json',costs)
    complete=all(stats[c]['completed']==len(tasks) for c in ['B2','B3'])
    comparison={'complete':complete,'paired_completed':sum(all(r[c+'_score']!='pending' for c in ['B2','B3']) for r in rows),
        'B3_gain_tasks':[r['task'] for r in rows if r['B2_score']==0 and r['B3_score']==1],
        'B3_loss_tasks':[r['task'] for r in rows if r['B2_score']==1 and r['B3_score']==0]}
    write(out/'comparison.json',comparison)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>B2/B3 pilot trajectories</h1><pre>'+html.escape(json.dumps(stats,indent=2))+'</pre>'+''.join(f'<p><a href="{r["task"]}.html">{r["task"]}</a>: B2 {r["B2_score"]}, B3 {r["B3_score"]}</p>' for r in rows))
    print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
