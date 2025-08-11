#Write a code so we can train HF model with adapters. The different models can be trained and in a loop use different data types.
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AdapterTrainer, TrainingArguments
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, AutoAdapterModel, AdapterConfig, PfeifferConfig
from transformers import EarlyStoppingCallback, DataCollatorWithPadding, EvalPrediction
from datasets import Dataset
from sklearn.metrics import accuracy_score
import transformers
import torch
import os,gc,time
from sklearn import metrics
from scipy.stats import pearsonr as pearsonr_scipy
import glob


# Argument parser setup
parser = argparse.ArgumentParser(description='Training script for HF model with adapter.')
parser.add_argument('--data_path', type=str, default = "../Dataset/combined_dataframe.pkl", help='Path to the PD file containing data.')
parser.add_argument('--grad_acc_steps', type=int, default=1, help='Number of gradient accumulation steps.')
parser.add_argument('--data_type', type=str, default='mo_trg', choices=['mo_trg', 'mo', 'DEP', 'POS'], help='Choose training data type.')
parser.add_argument('--model_path', default = "", type=str, help='Path to the folder containing the HF models.')
parser.add_argument('--output_dir', type=str, default='results/', help='Output directory for the trained model.')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and evaluation.')
#parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use. Set to -1 for CPU.')
parser.add_argument('--model_type', type=str, default='bert-large-cased', help='Choose between BERT, ROBERTA models.', choices=['bert-large-cased','roberta-large'], )


args = parser.parse_args()

# Load data from dataframe
df = pd.read_pickle(args.data_path)

#Create the list of POS or DEP tags from the dataset:
if args.data_type in ['DEP', 'POS']:
	TAGS = df["POS"].tolist()
	TAGS = " ".join(TAGS).split()
	TAGS = list(set(TAGS))

# Only sample those rows where the Time <250 seconds:
df = df[df['use_MO_TRG'] == 1]
# Split the data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=240)

def pearson_loss1(x, y):
	x_ = x - torch.mean(x, axis=-1, keepdim=True)
	y_ = y - torch.mean(y, axis=-1, keepdim=True)
	corr = torch.sum(x_ * y_, axis=-1) / torch.sqrt(torch.sum(torch.square(x_), axis=-1) * torch.sum(torch.square(y_), axis=-1) + 1e-6)
	return 1 - corr

def pearson_loss2(x,y):
	x_ = x - torch.mean(x, dim=-1, keepdim=True)
	y_ = y - torch.mean(y, dim=-1, keepdim=True)
	cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
	#p_correlation = torch.sum(1-cos(centered_1, centered_2))
	p_correlation = cos(x_, y_)
	return 1 - p_correlation

""" create compute_metrics function """
def compute_metrics(p):
	predictions, labels = p
	preds = predictions[:, 0]
	lossMAE = metrics.mean_absolute_error(preds, labels)
	lossPCC, pval = pearsonr_scipy(preds, labels)
	met = {"MAE": lossMAE, "PCC": lossPCC, "PCC_pval": pval}
	return met

convF = lambda p: torch.FloatTensor(p)

def create_dataset(dataframe, data_type):
	#Create the dataset. Combine the sentences together and use "time_avg_MO_TRG" as the time value of the pandas dataframe.
	sent = []
	time = []
	for i in range(dataframe.shape[0]):
		tmp = dataframe.iloc[i]
		#src = tmp["SRC"]
		mo = tmp["SRC"]
		trg = tmp["TRG"]
		pos = tmp["POS"]
		nEW = tmp["NumEditedWords"]
		#if data_type == 'src_mo':
		#	text = f"[SRC] {src} [MO] {mo}"
		if data_type == 'mo_trg':
			text = f"[MO] {mo} [TRG] {trg}"
		#elif data_type == 'src_mo_trg':
		#	text = f"[SRC] {src} [MO] {mo} [TRG] {trg}"
		elif data_type == 'mo':
			text = f"[MO] {mo}"
		elif data_type == 'DEP':
			text = str(nEW) + " - " + pos
		elif data_type == 'POS':
			text = pos
		sent.append(text)
		time.append(tmp["time_avg_MO_TRG"])
	data = pd.DataFrame({'text':sent, 'time':time})
	return Dataset.from_dict(data)

def tokenizeData(examples, tokenizer):
	#inputs = tokenizer(examples["text"], truncation=True, max_length = 510, padding=True)
	inputs = tokenizer(examples["text"], truncation=True, max_length = 510, padding=False)
	inputs['labels'] = examples['time']
	return inputs


dt = args.data_type

#Load the Models:
config = AutoConfig.from_pretrained(args.model_path+args.model_type, num_labels=1)
#model = AutoAdapterModel.from_pretrained(args.model_path+args.model_type, config=config)
model = AutoModelForSequenceClassification.from_pretrained(args.model_path+args.model_type, config=config)
tokenizer = AutoTokenizer.from_pretrained(args.model_path+args.model_type, model_max_length=512)
# Ensure padding token is set
if tokenizer.pad_token_id is None:
	raise ValueError("Padding token is not set.")
#Add the extra tokens to the tokenizer and mode vocabulary for [SRC] [TRG] [MO]
if dt not in ['DEP', 'POS']:
	additional_special_tokens = ["[SRC]", "[TRG]", "[MO]"]
else:
	additional_special_tokens = TAGS

special_tokens_dict = {"additional_special_tokens": additional_special_tokens}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))
#create adapter with classification head and activate
#adapters.init(model)
adap_name = args.model_type + "_" + dt
adap_config = PfeifferConfig()
model.add_adapter(adap_name, config=adap_config)
# Add a matching classification head
#model.add_classification_head(adap_name, num_labels=1)
# Activate the adapter
model.train_adapter(adap_name)
# Move the model to the selected device
#model.to(device)
#set training arguments
#Data Collator:
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
metric_for_best_model='eval_PCC'
learning_rate=1e-04
early_stopping_patience=10
early_stopping_threshold=.05
callbacks=[]
callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
# Define training arguments using command-line arguments
training_args = TrainingArguments(
	output_dir = dt + "_" + args.output_dir,
	num_train_epochs=args.num_epochs,
	per_device_train_batch_size=args.batch_size,
	per_device_eval_batch_size=args.batch_size,
	learning_rate=learning_rate,
	gradient_accumulation_steps=args.grad_acc_steps,
	#warmup_steps=100,
	weight_decay=0.01,
	logging_dir='./logs',
	logging_steps=200,
	#use_cpu = False,
	#fp16 = False,
	#eval_steps=500,
	#save_steps=300,
	#dataloader_pin_memory = False,
	#dataloader_num_workers = 6,
	evaluation_strategy='epoch',
	disable_tqdm=False,
	overwrite_output_dir=True,
	save_strategy='epoch',
	load_best_model_at_end=True,
	#remove_unused_columns=False,
	metric_for_best_model=metric_for_best_model,
	report_to='none',
	resume_from_checkpoint = True
	)

#Create the dataset and load it:
#create_dataset(dataframe, data_type)
ds_train = create_dataset(train_df, data_type = dt)
ds_val = create_dataset(val_df, data_type = dt)
train_dataset = ds_train.map(lambda x: tokenizeData(x, tokenizer), batched=True, num_proc = 6)
val_dataset = ds_val.map(lambda x: tokenizeData(x, tokenizer), batched=True, num_proc = 6)
trainer = AdapterTrainer(
	model=model,
	args=training_args,
	train_dataset=train_dataset,
	eval_dataset=val_dataset,
	compute_metrics=compute_metrics,
	data_collator=data_collator,
	callbacks=callbacks,
	)
# Train the model
#Check if the training crashed before and if it did load a checkpoint and resume training.
fCont = glob.glob(dt + "_" + args.output_dir+"/*")
if fCont:
	trainer.train(resume_from_checkpoint = True)
else:
	trainer.train()


eval_output = trainer.evaluate()
eval_metric_result = eval_output[metric_for_best_model]
evRes = pd.DataFrame({'metric':list(eval_output.keys()), 'value': list(eval_output.values())}, columns=['metric', 'value'])
evRes.to_pickle(args.model_type + "_" + dt + "_eval_metrics.pd")
""" set path for where to save the adapter """
adapter_save_path = dt + "_" + args.output_dir
""" save """
trainer.model.save_adapter(adapter_save_path, adap_name)
#pause for a while:
del model, tokenizer, trainer, train_dataset, val_dataset
gc.collect()
torch.cuda.empty_cache()
print("Sleeping for 2 minutes. Pause!\n")
print("Done with " + dt)
time.sleep(2*60)


