#Script to generate the estimated PEET for each model in the M2 files compared to multiple references.
#The model considers the best time to correct among multiple references for each sentence.

import glob
import pandas as pd
import pickle as p


F = glob.glob("M2/*.m2")
MO = glob.glob("MO/*")
REF = glob.glob("REF/*")
allRes = {}

def mainWO(fName):
	m2 = open(fName).read().strip().split("\n\n")
	# Do not apply edits with these error types
	skip = {"noop", "UNK", "Um"}
	SENT = []
	ANNOT = []
	for sent in m2:
		sent = sent.split("\n")
		cor_sent = sent[0].replace("S ","",1) # Ignore "S "
		SENT.append(cor_sent)
		edits = [t[2:] for t in sent[1:]]
		edits2 = []
		for edit in edits:
			tmpE = edit.split("|||")
			if not tmpE[1] in skip:# Ignore certain edits
				tmpEM = "|||".join(tmpE[:3])
				edits2.append(tmpEM)
		ANNOT.append(edits2)
	return SENT, ANNOT


def predictResults(dfT):
	checkColumns = ['R', 'M', 'U', 'NumWordsS', 'NumWordsT', 'NumEditedWords']
	X_test = dfT[checkColumns]
	filename = 'modelLR.sav'
	loaded_model = p.load(open(filename, 'rb'))
	y_pred = loaded_model.predict(X_test)
	return y_pred


for f in F:
	moTxt, trgTxt = f.split("/")[-1].split(".")[0].split("_")
	moTxt = open("MO/"+moTxt,"r").read().strip().split("\n")
	trgTxt = open("REF/"+trgTxt,"r").read().strip().split("\n")
	s,a = mainWO(f)
	a2 = []
	for i in a:
		if i:
			a2.append(i)
		else:
			a2.append(["0 0|||NONE|||NONE"])
	assert(len(moTxt) == len(trgTxt) == len(a2))
	dataL = {}
	dataL["R"] = []
	dataL["M"] = []
	dataL["U"] = []
	dataL["NumWordsS"] = []
	dataL["NumWordsT"] = []
	dataL["NumEditedWords"] = []
	L = len(a2)
	for i in range(L):
		lbl = a2[i]
		tmpDict = {"M":0,"R":0,"U":0}
		#present ones:
		present = []
		for j in lbl:
			val = j.split("|||")
			assert(len(val) == 3)
			cntr = [int(k) for k in val[0].split()]
			assert(len(cntr) == 2)
			cntr = cntr[1] - cntr[0]
			if cntr == 0:
				cntr = 1
			assert(cntr > 0)
			present.append([val[1], cntr, val[2]])
		for k in present:
			if k[0][0] in tmpDict:
				tmpDict[k[0][0]] += k[1]
		for k in tmpDict:
			dataL[k].append(tmpDict[k])
		#Num words in source and target/edit
		dataL["NumWordsS"].append(len(moTxt[i].split()))
		dataL["NumWordsT"].append(len(trgTxt[i].split()))
		#Num of edited words - deleted or added or modified
		tmp = 0
		for k in present:
			if not k[0].lower() == "none":
				tmp += k[1]
		dataL["NumEditedWords"].append(tmp)
	#Now predict the score for each item!
	dfTmp = pd.DataFrame.from_dict(dataL)
	tmpRes = predictResults(dfTmp)
	allRes[f.split("/")[-1].split(".")[0]] = tmpRes


finalRes = {}
tmpRes = {}

for f in allRes:
	nm = f.split("_")
	if not nm[0] in tmpRes:
		tmpRes[nm[0]] = []
	tmpRes[nm[0]].append(allRes[f])

for f in tmpRes:
	finalRes[f] = []
	assert(len(tmpRes[f]) == len(REF))
	L = len(tmpRes[f][0])
	for j in range(L):
		tmp = []
		for i in tmpRes[f]:
			tmp.append(i[j])
		finalRes[f].append(min(tmp))


tmp = []

#Check count of all key-values in finalRes is the same.

for t in finalRes:
	tmpCnt = len(finalRes[t])
	break

for t in finalRes:
	assert(len(finalRes[t]) == tmpCnt)
	netWER = sum(finalRes[t])
	tmp.append([t, netWER/tmpCnt])

tmp.sort(key=lambda x: x[1])

print("The sentence average PEET for each model is:")
#print(tmp)
for i in tmp:
	print(i[0] + " : " + str(round(i[1],4)))


print("Total sentences processed: " + str(tmpCnt))