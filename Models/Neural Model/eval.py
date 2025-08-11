import pandas as pd
import glob
F = glob.glob("*.pd")

#['bert-base-cased_src_mo_eval_metrics.pd', 'bert-base-cased_src_mo_trg_eval_metrics.pd', 'bert-base-cased_mo_eval_metrics.pd', 'bert-base-cased_mo_trg_eval_metrics.pd']

res = ["eval_MAE", "eval_PCC", "eval_PCC_pval", "epoch"]

def printRes(nm):
	df = pd.read_pickle(nm)
	model = nm.split("_")[0]
	data_type = nm.replace("_eval_metrics.pd", "")
	data_type = "_".join(data_type.split("_")[1:])
	print("Model: ", model)
	print("Data type: ", data_type)
	print(df[df['metric'].isin(res)])


for f in F:
	printRes(f)