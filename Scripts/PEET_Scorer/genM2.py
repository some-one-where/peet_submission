import subprocess
import glob
import multiprocessing as mp

'''
errant_parallel -orig src.txt -cor trg0.txt -out M2Files/src_trg0.m2
errant_parallel -orig src.txt -cor trg1.txt -out M2Files/src_trg1.m2
'''

TRG = glob.glob("REF/*")

def getM2File(x):
	mTmp, tTmp = x
	mName = mTmp.split("/")[-1]
	tName = tTmp.split("/")[-1]#.split(".")[0]
	outName = mName+"_"+tName+".m2"
	cmnd = "errant_parallel -orig "+ mTmp +" -cor "+tTmp+" -out M2/"+outName
	tmp = subprocess.run(cmnd.split())
	print("Done! "+mName+" "+tName)


if __name__ == '__main__':
	MO = glob.glob("MO/*")
	pairings = []
	for t in TRG:
		for m in MO:
			pairings.append([m,t])
	cCount = mp.cpu_count() - 2
	p = mp.Pool(cCount)
	p.map(getM2File, pairings)


