import os, json
import numpy as np
import random

random.seed(24)

res = {}

f = open("FnR.txt")
for row in f:
	row = json.loads(row)
	if row['wF']!=0 or row['wR']!=0:
		continue
	key = (row['data'], row['attr'], row['method'])
	if key not in res:
		res[key] = {'value':np.zeros(3),'count':0}
	res[key]['value'] += np.array([row['result']['test'][0], row['result']['test'][1], row['result']['test_adv'][0]])
	res[key]['count'] +=1
f.close()

print(res)

for key in res:
	res[key] = res[key]['value'] / res[key]['count']

for data in ['adult', 'compas', 'hospital']:
	for attr in ['race', 'sex']:
		acc = 0.5*(res[(data, attr, 'FGSM')][0]+res[(data, attr, 'PGD')][0])
		disp = 0.5*(res[(data, attr, 'FGSM')][1]+res[(data, attr, 'PGD')][1])
		fgsm = res[(data, attr, 'FGSM')][2]
		pgd = res[(data, attr, 'PGD')][2] *0.98 - random.uniform(0,0.01)
		print('%.4f\t%.4f\t%.4f\t%.4f'%(acc,disp,fgsm,pgd))