import os
import pickle
import numpy as np
import pandas as pd

def do(data, attr, method, req):

	n_advexp=0
	n_flips=0
	count=0

	orig = pd.read_csv(f'../../data/{data}_train.csv').to_numpy()

	fnames = os.listdir('./')
	for fname in fnames:
		if '.pre' not in fname:
			continue
		setting = fname.split('_')
		if (setting[0], setting[1], setting[2]) != (data, attr, method):
			continue
		with open('./'+fname, 'rb') as handle:
			res = pickle.load(handle)
			dR = res['dR']
			dF = res['dF']
			key = (dR, dF)
			# print(data, attr, method, key)
			if key!=req:
				continue
			
			n_advexp += res['downstream_train']['X'].shape[0] - orig.shape[0]
			n_flips += (np.array(res['downstream_train']['y']).reshape(-1)[:orig.shape[0]]!=orig[:,-1]).sum()
			count+=1

	n_advexp/=count
	n_flips/=count

	print((n_advexp, n_advexp/orig.shape[0]), (n_flips, n_flips/orig.shape[0]))
			
			
if __name__=='__main__':

	reqs={
		'adult': (0.6, 0.03),
		'compas': (0.55, 0.05),
		'hospital': (0.45, 0.015),
	}

	for data in sorted(reqs.keys()):
		do(data,'race','FGSM',reqs[data])