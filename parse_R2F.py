import json
import numpy as np

data = {}

f=open('./result/existings/R2F.txt')
for line in f:
	obj=json.loads(line)
	combo=(obj['data'],obj['attr'])
	if combo not in data:
		data[combo]={'count':0,}
	if (obj['method'], obj['wR']) not in data[combo]:
		data[combo][(obj['method'], obj['wR'])]=np.array([0.,0.])
	data[combo][(obj['method'], obj['wR'])]+=np.array(obj['test'])
f.close()

for combo in data:
	for item in data[combo]:
		data[combo][item]/=50

for method in ['FGSM', 'PGD']:
	for combo in sorted(data.keys()):
		if combo=='count':
			continue
		for wR in [0.01,0.05,0.1,0.5,1.0]:
			change = data[combo][(method,wR)]-data[combo][('None',0.1)]
			print(combo, method, wR, round(change[1],4))