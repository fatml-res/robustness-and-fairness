import numpy as np
import json

fairness_list = [
	(0.0, None), 
	(1.0, 'fairness_reweighing'),
	(0.2, 'fairness_disparate'),
	(0.4, 'fairness_disparate'),
	(0.6, 'fairness_disparate'),
	(0.8, 'fairness_disparate'),
	(1.0, 'fairness_disparate'),
]

robustness_list = [
	0.0, 0.01, 0.05, 0.1, 0.5, 1.0
]

res={}
count={}

f=open('FnR.txt','r')
for row in f:
	obj=json.loads(row)
	data = obj['data']
	attr = obj['attr']
	method = obj['method']
	wR = obj['wR']
	wF = obj['wF']
	func = obj['func']
	if wF == 0.0:
		func = None
	if wF != 0.0:
		continue
	setting = (data, attr)
	if setting not in res:
		res[setting]={}
		count[setting]={}
		res[setting]['FGSM']=np.zeros(len(robustness_list))
		res[setting]['PGD']=np.zeros(len(robustness_list))
		count[setting]['FGSM']=np.zeros(len(robustness_list))
		count[setting]['PGD']=np.zeros(len(robustness_list))
	res[setting][method][robustness_list.index(wR)] += obj['result']['test'][1]
	count[setting][method][robustness_list.index(wR)]+=1
f.close()


for item in res:
	for method in res[item]:
		for j in range(0,len(res[item][method])):
			res[item][method][j]/=count[item][method][j]

# print(res)


for item in res:
	for method in res[item]:
		res[item][method]-=res[item][method][0]



for item in sorted(res.keys()):
	# print(item)
	for j in range(1,len(res[item]['FGSM'])):
		print('%.04f'%round(res[item]['FGSM'][j],4),end='')
		print(' & ',end='')
	for j in range(1,len(res[item]['PGD'])):
		print('%.04f'%round(res[item]['PGD'][j],4),end='')
		print(' & ',end='')
	print()
