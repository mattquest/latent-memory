"""CPU-only independent receipt audit. Run from repository root after focused completion."""
import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path('runs/focused-ablation')
REPORT = Path('reports/2026-09-11/focused')
MATRIX = json.loads(Path('configs/focused-ablation.json').read_text())
receipt = {'scope': 'posthoc descriptive audit; no generation or causal claims',
           'jobs': [], 'groups': [], 'checks': [], 'e4_e5_overlap': []}
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def check(value, label):
    assert value, label
    receipt['checks'].append(label)
def key(r):
    c = r['case']
    return (r['example_id'], c['experiment'], c['arm'], c['hops'], float(c['bridge_ratio']),
            c['precision'], c['repeat'], c['control'], c['update_policy'])
def percentile(values, q):
    values = sorted(values)
    x = (len(values)-1)*q
    lo = int(x)
    return values[lo] + (values[min(lo+1,len(values)-1)]-values[lo])*(x-lo)
allrows, e2ids = [], []
for job in MATRIX['jobs']:
    name = job['name']
    config = {**MATRIX['defaults'], **job['config']}
    directory = ROOT/name
    manifest = json.loads((directory/'manifest.json').read_text())
    data = Path(config['dataset'])
    examples = [json.loads(x) for x in data.read_text().splitlines()][:config['limit']]
    rows = [json.loads(x) for x in (directory/'results.jsonl').read_text().splitlines()]
    check(sha(data) == manifest['identity']['dataset_sha256'], name+': exact dataset identity')
    check(all(r['status']=='ok' for r in rows), name+': no error records')
    check(len({r['id'] for r in rows})==len(rows), name+': unique result IDs')
    expected = set()
    for e in examples:
        for experiment in config['experiments']:
            if experiment == 'E2':
                cases = [(a,h,.2,p) for h in config['hops'] for a,p in
                         [('text','bf16'),('latent','bf16'),('latent','int8'),('direct_text','bf16')]]
            elif experiment == 'E4':
                cases = [('latent',3,r,'bf16') for r in [0,.1,.2,1]]
            elif experiment == 'E5':
                cases = [('latent',3,.2,p) for p in ['bf16','int8','int4']]
            for a,h,r,p in cases:
                expected.add((e['id'],experiment,a,h,float(r),p,0,'correct','all'))
    check(set(map(key,rows))==expected and len(rows)==len(expected), name+': exact requested grid')
    check(all(r['counts']['host_output_tokens']<=config['max_answer_tokens'] for r in rows), name+': final output bounds')
    check(all(r['counts']['max_prefix_tokens']<=8192 for r in rows), name+': model prefix bounds')
    check(all(r['counts']['evidence_tokens']<=6144 for r in rows), name+': evidence bounds')
    receipt['jobs'].append({'job':name, 'examples':len(examples), 'conditions':len(rows),
                           'results_sha256':sha(directory/'results.jsonl'),
                           'dataset_sha256':sha(data), 'manifest_sha256':sha(directory/'manifest.json')})
    allrows.extend(rows)
    groups = defaultdict(list)
    for r in rows:
        c = r['case']
        groups[(c['experiment'],c['arm'],c['hops'],float(c['bridge_ratio']),c['precision'])].append(r)
    for k, g in sorted(groups.items()):
        entry = {'job':name, 'experiment':k[0], 'arm':k[1], 'nominal_hops':k[2], 'ratio':k[3], 'precision':k[4],
                 'actual_hops':sorted({r['evidence_hops_executed'] for r in g}), 'n':len(g),
                 'correct':sum(r['scores']['exact_match'] for r in g), 'f1':statistics.mean(r['scores']['f1'] for r in g),
                 'warm_p50_ms':percentile([r['latency_ms'] for r in g],.5),
                 'warm_p95_ms':percentile([r['latency_ms'] for r in g],.95),
                 'unknown':sum(r['prediction'].strip().casefold()=='unknown' for r in g),
                 'final_output_capped':sum(r['generation_stop']=='max_tokens' for r in g),
                 'selected_document_truncated':sum(bool(set(r['evidence_ids']) & set(r['truncated_document_ids'])) for r in g),
                 'all_support_document_ids':sum(r['support_recall']==1 for r in g),
                 'mean_internal_tokens':statistics.mean(r['counts']['internal_decoded_tokens'] for r in g),
                 'mean_host_tokens':statistics.mean(r['counts']['host_output_tokens'] for r in g),
                 'median_union_ingest_ms':statistics.median(r['cold_ingest_ms'] for r in g),
                 'bridge_recomputed_tokens_range':[min(r['counts']['bridge_recomputed_tokens'] for r in g),max(r['counts']['bridge_recomputed_tokens'] for r in g)],
                 'evidence_token_range':[min(r['counts']['evidence_tokens'] for r in g),max(r['counts']['evidence_tokens'] for r in g)]}
        if k[1]=='latent':
            entry['cache_bytes_per_token'] = sorted({r['cache']['nbytes']/r['cache']['seq_len'] for r in g})
        receipt['groups'].append(entry)
    if config['experiments']==['E2']:
        e2ids.extend(e['id'] for e in examples)
        bypair = defaultdict(list)
        for r in rows:
            bypair[(r['example_id'],r['case']['hops'])].append(r)
        check(all(len({tuple(r['evidence_ids']) for r in arms})==1 for arms in bypair.values()), name+': matched selected ID order')
        check(all(len({r['evidence_hops_executed'] for r in arms})==1 for arms in bypair.values()), name+': matched actual hops')
        check(all(len({tuple(bypair[(e['id'],h)][0]['evidence_ids']) for h in config['hops']})==len(config['hops']) for e in examples), name+': unique selected schedules')
    else:
        byid = defaultdict(dict)
        for r in rows:
            byid[r['example_id']][(r['case']['experiment'],float(r['case']['bridge_ratio']),r['case']['precision'])] = r
        check(all(len({tuple(r['evidence_ids']) for r in g.values()})==1 for g in byid.values()), name+': matched E4/E5 selected ID order')
        check(all(len({r['cache']['seq_len'] for r in g.values()})==1 for g in byid.values()), name+': matched E4/E5 prefix lengths')
        for eid,g in byid.items():
            a,b = g[('E4',.2,'bf16')],g[('E5',.2,'bf16')]
            check(a['output_token_ids']==b['output_token_ids'] and a['scores']==b['scores'], name+': repeated BF16 output '+eid)
            receipt['e4_e5_overlap'].append({'job':name,'example_id':eid,'equal_output_tokens':True})
        check(all(g[('E4',1.,'bf16')]['counts']['bridge_recomputed_tokens']==g[('E4',1.,'bf16')]['cache']['seq_len'] for g in byid.values()), name+': full recompute counts')
        check(all(g[('E4',0.,'bf16')]['counts']['bridge_recomputed_tokens']==0 for g in byid.values()), name+': zero recompute counts')
        for precision, factor in [('int8',.53125),('int4',.28125)]:
            check(all(g[('E5',.2,precision)]['cache']['nbytes']/g[('E5',.2,'bf16')]['cache']['nbytes']==factor for g in byid.values()), name+': exact packed '+precision+' fraction')
check(len(allrows)==660, 'exactly 660 successful unique conditions')
check(len(e2ids)==16 and len(set(e2ids))==16, 'sixteen disjoint E2 questions')
artifacts = json.loads((REPORT/'artifact-manifest.json').read_text())['artifacts']
for r in artifacts:
    p = REPORT/r['artifact']
    check(sha(p)==r['artifact_sha256'], 'artifact digest '+r['artifact'])
    if 'source_sha256' in r:
        raw = gzip.decompress(p.read_bytes()) if r.get('compression')=='gzip' else p.read_bytes()
        check(hashlib.sha256(raw).hexdigest()==r['source_sha256'], 'archived source digest '+r['artifact'])
receipt['archived_receipts_verified'] = len(artifacts)
receipt['guard'] = json.loads((ROOT/'guard-status.json').read_text())
check(receipt['guard']['status']=='complete' and receipt['guard']['returncode']==0, 'successful resource guard')
receipt['checks_passed'] = len(receipt['checks'])
receipt['final_output_capped_total'] = sum(r['generation_stop']=='max_tokens' for r in allrows)
receipt['audit_source_sha256'] = sha(Path(__file__))
Path('runs/focused-final-audit.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ['checks','groups','e4_e5_overlap']},indent=2))
print('PUBLIC',json.dumps([g for g in receipt['groups'] if g['experiment']!='E2'],indent=2))
