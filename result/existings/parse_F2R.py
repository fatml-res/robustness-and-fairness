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
	if wR > 0:
		continue
	if wF == 0.0:
		func = None
	setting = (data, attr, method)
	if setting not in res:
		res[setting]=np.zeros(len(fairness_list))
		count[setting]=np.zeros(len(fairness_list))
	res[setting][fairness_list.index((wF, func))]+=obj['result']['test_adv'][0]
	count[setting][fairness_list.index((wF, func))]+=1
f.close()

# print(count)

for item in res:
	for j in range(0,len(res[item])):
		res[item][j]/=count[item][j]
	res[item]-=res[item][0]

for item in sorted(res.keys()):
	print(item)
	for j in range(1,len(res[item])):
		print('%.04f'%round(res[item][j],4),end='')
		if j==len(res[item])-1:
			print(' \\\\')
		else:
			print(' & ',end='')