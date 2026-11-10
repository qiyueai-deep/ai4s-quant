"""Add uncertainty and bias checks to the rare-event comparison."""
import argparse, json
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input',default='results/rare_event.tsv'); ap.add_argument('--output',default='results/rare_event_uncertainty.json'); a=ap.parse_args()
 row=Path(a.input).read_text(encoding='utf-8').splitlines()[1].split('\t'); p_true,var_m,var_d,reduction=map(float,row)
 # Variances in the source experiment are variances across 20 repeated estimates.
 n_rep=20; se_m=(var_m/n_rep)**0.5; se_d=(var_d/n_rep)**0.5
 out={'p_true':p_true,'variance_mcmc':var_m,'variance_diffusion':var_d,'variance_reduction':reduction,'repeats':n_rep,'approx_se_of_repeat_mean':{'mcmc':se_m,'diffusion':se_d},'relative_bias_requires_raw_estimates':True,'effective_sample_size_requires_chain_trace':True}
 Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
