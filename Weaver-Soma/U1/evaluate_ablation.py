from pathlib import Path
import json,csv,torch,numpy as np
from sklearn.metrics import average_precision_score
from train_unified import make_data
from unified_model import WeaverSOMAUnified
PATH=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2)
    obj=torch.load(PATH/'results/weaver_soma_unified.pt',map_location='cpu',weights_only=True)
    model=WeaverSOMAUnified();model.load_state_dict(obj['weights']);model.eval()
    rows=[]
    for scenario,seed in [('normal',10000),('changed',10001),('extreme',10002)]:
        b=make_data(2200,seed,scenario)
        for ablation in ['none','workspace','memory','lexicon']:
            probs=[];wp=[];cp=[];summ=[]
            with torch.inference_mode():
                for i in range(0,2200,512):
                    s={k:v[i:i+512] for k,v in b.items()}
                    o=model(s['obs'],s['quality'],s['context'],s['executed_action'],ablate=ablation)
                    probs.append(o['risk_logits'].sigmoid());wp.append(o['workspace_logits'].argmax(-1));cp.append(o['compute_logits'].argmax(-1));summ.append(o['episodic_summary'])
            probs=torch.cat(probs).numpy()
            r=float(average_precision_score(b['risk'].numpy().flatten(),probs.flatten()))
            rows.append({'scenario':scenario,'ablation':ablation,'risk_ap':r,
                         'attention_accuracy':float(np.mean(torch.cat(wp).numpy()==b['attention'].numpy())),
                         'route_accuracy':float(np.mean(torch.cat(cp).numpy()==b['route'].numpy())),
                         'memory_mae':float(np.abs(torch.cat(summ).numpy()-b['summary'].numpy()).mean())})
    with (PATH/'results/module_ablations.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
