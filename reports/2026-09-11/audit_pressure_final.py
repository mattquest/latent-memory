"""CPU-only independent reconciliation of saved pressure records; no model loads."""
import argparse, collections, gzip, hashlib, json, math, re, statistics, string
from pathlib import Path
from tokenizers import Tokenizer
from eval.adaptive_experiments import PROLOGUE
from eval.retrieval import STOPWORDS


def sha(raw): return hashlib.sha256(raw).hexdigest()
def jsha(value): return sha(json.dumps(value).encode())
def rows(raw): return [json.loads(line) for line in raw.splitlines() if line.strip()]
def pct(values, q):
    a=sorted(values); x=(len(a)-1)*q; i=int(x)
    return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
def terms(text): return [w for w in re.findall(r'[a-z0-9]+',text.lower()) if w not in STOPWORDS and len(w)>1]
def scores(prediction, answers):
    def norm(s): return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s.lower().translate(str.maketrans('','',string.punctuation))).split())
    p=norm(prediction); em=f1=0
    for answer in answers:
        a=norm(answer); equal=float(p==a); em=max(em,equal); pp,aa=p.split(),a.split()
        f=equal if p in {'yes','no','noanswer'} or a in {'yes','no','noanswer'} else float(pp==aa) if not pp or not aa else 2*sum((collections.Counter(pp)&collections.Counter(aa)).values())/(len(pp)+len(aa))
        f1=max(f1,f)
    return {'exact_match':em,'f1':f1}


def audit(run_dir, output):
    guard=json.loads((run_dir/'guard-status.json').read_text())
    summary=json.loads((run_dir/'summary.json').read_text())
    assert guard['status']=='complete' and guard['returncode']==0 and summary['status']=='complete'
    manifest_raw=(run_dir/'manifest.json').read_bytes(); manifest=json.loads(manifest_raw); identity=manifest['identity']; cfg=identity['config']
    failures=[]
    def check(ok,kind,example=None):
        if not ok: failures.append({'kind':kind,'example_id':example})
    archives={}
    for receipt in manifest['archives']:
        packed=(run_dir/receipt['artifact']).read_bytes(); raw=gzip.decompress(packed)
        check(sha(packed)==receipt['artifact_sha256'] and sha(raw)==receipt['source_sha256'],'archived_bytes')
        archives[receipt['artifact']]=raw
    for key,name in [('dataset','questions.jsonl.gz'),('corpus','corpus.jsonl.gz'),('source_results','source-results.jsonl.gz')]:
        check(sha(archives[name])==identity[key+'_sha256'],'identity_'+key)
    for name,expected in identity['source_sha256'].items(): check(sha(Path(name).read_bytes())==expected,'generation_source_hash_'+name)
    source_dir=Path(cfg['source_dir']); check(sha((source_dir/'manifest.json').read_bytes())==identity['source_manifest_sha256'],'source_manifest_identity')
    questions={r['id']:r for r in rows(archives['questions.jsonl.gz'])}; corpus={r['id']:r for r in rows(archives['corpus.jsonl.gz'])}
    sources={r['id']:r for r in rows(archives['source-results.jsonl.gz'])}
    controllers={r['example_id']:r for r in sources.values() if r.get('status')=='ok' and r['case']=={'arm':'iterative_text','max_rounds':cfg['source_rounds']}}
    check(set(controllers)==set(questions) and len(questions)==cfg['expected_questions'],'source_question_cohort')
    raw=(run_dir/'results.jsonl').read_bytes(); ledger=rows(raw); latest={r['id']:r for r in ledger}; successful=[r for r in latest.values() if r['status']=='ok']
    grouped=collections.defaultdict(dict)
    for r in successful: grouped[r['example_id']][r['arm']]=r
    arms=['retained_text','retained_kv_bf16','retained_kv_int8','rolling_summary']
    check(len(successful)==len(questions)*len(arms) and len(latest)==len(successful),'complete_condition_count')
    tokenizer_path=Path('models/Qwen3-8B/tokenizer.json'); tokenizer=Tokenizer.from_file(str(tokenizer_path))
    enc=lambda s:tokenizer.encode(s,add_special_tokens=False).ids
    dec=lambda t:tokenizer.decode(t,skip_special_tokens=True)
    start,end=enc('\n<retained_memory>\n'),enc('\n</retained_memory>\n'); prologue=enc(PROLOGUE)
    source_cfg=manifest['source_manifest']['identity']['config']; fragments={}
    for key in {key for source in controllers.values() for key in source['evidence_ids']}:
        doc=corpus[key]; full=enc(f"\n<document>\n{doc.get('title','')}\n{doc['text']}\n</document>\n")
        cap=full[:source_cfg['max_document_tokens']]
        fragments[key]={'tokens':cap,'full_tokens':len(full),'counts':collections.Counter(terms(dec(cap)))}
    overflow=[]; support_rows=[]; max_component_error=0
    for qid,q in questions.items():
        source=controllers[qid]; group=grouped[qid]; support=set(q['supporting_context_ids']); total=sum(len(fragments[k]['tokens']) for k in source['evidence_ids'])
        check(set(group)==set(arms),'all_arms',qid)
        if total>cfg['evidence_budget']: overflow.append(qid)
        arrived=[]; independent_retention=[]
        for event in (event for event in source['trace'] if event['event']=='retrieval'):
            arrived.extend(event['new_document_ids']); tokens=prologue+[t for key in arrived for t in fragments[key]['tokens']]
            check(arrived==event['accumulated_document_ids'] and jsha(tokens)==event['evidence_token_ids_sha256'],'source_tokens',qid)
            lengths={key:sum(fragments[key]['counts'].values()) for key in arrived}; average=sum(lengths.values())/len(arrived) or 1
            query=sorted(set(terms(q['question']))); df={term:sum(term in fragments[k]['counts'] for k in arrived) for term in query}
            ranked_scores={}
            for key in arrived:
                score=0
                for term in query:
                    f=fragments[key]['counts'].get(term,0)
                    if f: score+=math.log(1+(len(arrived)-df[term]+.5)/(df[term]+.5))*f*2.5/(f+1.5*(.25+.75*lengths[key]/average))
                ranked_scores[key]=score
            chosen=[]; used=0
            for key in sorted(arrived,key=lambda k:(-ranked_scores[k],k)):
                n=len(fragments[key]['tokens'])
                if used+n<=cfg['evidence_budget']: chosen.append(key); used+=n
            expected=[key for key in arrived if key in set(chosen)]; independent_retention.append(expected)
        for arm,r in group.items():
            check(r['id']==jsha([identity['protocol'],qid,arm,cfg['evidence_budget'],cfg['summary_budget']])[:24],'case_id',qid)
            check(r['question']==q['question'] and r['answers']==q['answers'] and r['scores']==scores(r['prediction'],q['answers']),'question_answer_scores',qid)
            check(r['raw_output']==dec(r['output_token_ids']),'output_tokens',qid)
            check(r['source_result_id']==source['id'] and r['source_trace_sha256']==sha(json.dumps(source['trace'],sort_keys=True).encode()),'controller_trace',qid)
            check(r['cumulative_source_tokens']==total and r['trace_overflow']==(total>cfg['evidence_budget']),'overflow_tokens',qid)
            for out,key in [('source_controller_ms','controller_ms'),('source_retrieval_ms','retrieval_ms')]: check(r[out]==source['component_ms'].get(key,0),'source_cost',qid)
            error=abs(r['replay_cold_ms']-sum(r['component_ms'].values())); max_component_error=max(max_component_error,error)
            check(error<1e-6,'component_cost_sum',qid)
            check(math.isclose(r['replay_plus_source_controller_and_retrieval_ms'],r['replay_cold_ms']+r['source_controller_ms']+r['source_retrieval_ms'],abs_tol=1e-7),'replay_accounting_sum',qid)
            memory=[]
            if arm=='rolling_summary' and r['summary_trace']:
                for entry in r['summary_trace']: check(jsha(entry['memory_token_ids'])==entry['memory_token_ids_sha256'] and dec(entry['memory_token_ids'])==entry['memory_text'],'summary_tokens',qid)
                memory=start+r['summary_trace'][-1]['memory_token_ids']+end
                check(len(memory)<=cfg['summary_budget'],'summary_budget',qid)
            evidence=memory+[t for key in r['retained_document_ids'] for t in fragments[key]['tokens']]
            check(jsha(evidence)==r['retained_evidence_token_ids_sha256'] and len(evidence)==r['retained_evidence_tokens'],'final_evidence_tokens',qid)
            check(len(evidence)<=cfg['evidence_budget'],'evidence_budget',qid)
            check(r['retained_truncated_document_ids']==[key for key in r['retained_document_ids'] if fragments[key]['full_tokens']>len(fragments[key]['tokens'])],'truncation_ids',qid)
            for label,ids in [('retained_original_support_id_coverage',r['retained_document_ids']),('summarized_source_support_id_coverage',r['summarized_source_document_ids'])]: check(r[label]==len(support&set(ids))/len(support),'support_coverage',qid)
            for call in r['model_calls']:
                check(call['evidence_input_tokens']<=cfg['evidence_budget'] and call['total_context_reservation_tokens']==sum(call[k] for k in ('evidence_input_tokens','fixed_prompt_question_tokens','output_reservation_tokens')) and call['total_context_reservation_tokens']<=cfg['max_model_context'],'call_budget',qid)
            if arm!='rolling_summary': check([e['retained_document_ids'] for e in r['retention_trace']]==independent_retention,'independent_bm25_retention',qid)
        check(len({jsha([group[a]['retained_document_ids'],group[a]['retained_evidence_token_ids_sha256'],group[a]['retention_trace']]) for a in arms[:3]})==1,'matched_text_kv_retention',qid)
        support_rows.append({'example_id':qid,'trace_overflow':total>cfg['evidence_budget'],'source_support_coverage':len(support&set(source['evidence_ids']))/len(support),'retained_support_coverage':group['retained_text']['retained_original_support_id_coverage'],'cumulative_tokens':total})
    metrics=[]; pairs=[]
    for subset,ids in [('all',list(questions)),('trace_overflow',overflow)]:
        for arm in arms:
            rs=[grouped[q][arm] for q in ids]
            metrics.append({'arm':arm,'subset':subset,'n':len(rs),'correct':sum(r['scores']['exact_match'] for r in rs),'em':statistics.mean(r['scores']['exact_match'] for r in rs),'f1':statistics.mean(r['scores']['f1'] for r in rs),'replay_p50_ms':pct([r['replay_cold_ms'] for r in rs],.5),'replay_p95_ms':pct([r['replay_cold_ms'] for r in rs],.95),'accounting_sum_p50_ms':pct([r['replay_plus_source_controller_and_retrieval_ms'] for r in rs],.5),'source_controller_mean_ms':statistics.mean(r['source_controller_ms'] for r in rs),'source_retrieval_mean_ms':statistics.mean(r['source_retrieval_ms'] for r in rs),'retained_support_coverage':statistics.mean(r['retained_original_support_id_coverage'] for r in rs),'summarized_source_support_coverage':statistics.mean(r['summarized_source_support_id_coverage'] for r in rs),'mean_retained_tokens':statistics.mean(r['retained_evidence_tokens'] for r in rs),'examples_with_retained_truncation':sum(bool(r['retained_truncated_document_ids']) for r in rs),'summary_generation_calls':sum(len(r['summary_trace']) for r in rs)})
            metrics[-1].update(summary_examples=sum(bool(r['summary_trace']) for r in rs),
                              max_evidence_input_tokens=max(c['evidence_input_tokens'] for r in rs for c in r['model_calls']),
                              max_total_context_reservation_tokens=max(c['total_context_reservation_tokens'] for r in rs for c in r['model_calls']),
                              mean_component_ms={k:statistics.mean(r['component_ms'].get(k,0) for r in rs) for k in sorted(set().union(*(r['component_ms'] for r in rs)))})
            if arm!='retained_text':
                controls=[grouped[q]['retained_text'] for q in ids]; diffs=[r['scores']['exact_match']-c['scores']['exact_match'] for r,c in zip(rs,controls)]
                pairs.append({'arm':arm,'reference':'retained_text','subset':subset,'n':len(rs),'em_difference':statistics.mean(diffs),'target_only_correct':sum(d>0 for d in diffs),'reference_only_correct':sum(d<0 for d in diffs),'mean_replay_difference_ms':statistics.mean(r['replay_cold_ms']-c['replay_cold_ms'] for r,c in zip(rs,controls)),'median_paired_replay_ratio':statistics.median(r['replay_cold_ms']/c['replay_cold_ms'] for r,c in zip(rs,controls))})
    result={'status':'PASS' if not failures else 'FAIL','failures':failures,'rows':len(ledger),'unique_ok':len(successful),'historical_error_records':sum(r['status']!='ok' for r in ledger),'n_questions':len(questions),'n_overflow':len(overflow),'overflow_ids':overflow,'max_component_reconciliation_error_ms':max_component_error,'metrics':metrics,'pairs':pairs,'support_rows':support_rows,'guard':guard,'identity':identity,'audit_source_sha256':sha(Path(__file__).read_bytes()),'run_manifest_sha256':sha(manifest_raw),'results_sha256':sha(raw),'tokenizer_sha256':sha(tokenizer_path.read_bytes())}
    output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({k:result[k] for k in ('status','failures','rows','unique_ok','n_questions','n_overflow','max_component_reconciliation_error_ms')}))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--run-dir',type=Path,default=Path('runs/context-pressure')); parser.add_argument('--output',type=Path,default=Path('runs/pressure-final-audit.json')); args=parser.parse_args(); audit(args.run_dir,args.output)
