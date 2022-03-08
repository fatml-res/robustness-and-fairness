import random
import numpy as np
import json

random.seed(24)

res = {}

def run(data, attr, method):

	f=open('FnR.txt','r')
	for row in f:
		obj=json.loads(row)

		if (obj['data'], obj['attr'], obj['method'])!=(data, attr, method):
			continue

		wR = obj['wR']
		wF = obj['wF']
		func = obj['func']
		if wF == 0.0:
			func = None
		param = (wR, wF, func)
		if param not in res:
			res[param]={
				'data': np.zeros(3),
				'count': 0
			}
		res[param]['data']+=np.array([obj['result']['test'][0], obj['result']['test_adv'][0], obj['result']['test'][1]])
		res[param]['count']+=1
	f.close()

	for item in res:
		res[item] = res[item]['data']/res[item]['count']

	fairness_list = [
		(0.0, None), 
		(1.0, 'fairness_reweighing'),
		# (0.2, 'fairness_disparate'),
		# (0.4, 'fairness_disparate'),
		# (0.6, 'fairness_disparate'),
		# (0.8, 'fairness_disparate'),
		(1.0, 'fairness_disparate'),
	]

	robustness_list = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]

	# for ftem in fairness_list:
	# 	for rtem in robustness_list:
	# 		param = (rtem, ftem[0], ftem[1])
	# 		A = res[param][0]
	# 		R = 1.0-res[param][1]
	# 		F = res[param][2]
	# 		print('(%.04f, %.04f, %.04f)'%(A, R, F),end='\t')
	# 	print()

	for ftem in fairness_list:
		for rtem in robustness_list:
			param = (rtem, ftem[0], ftem[1])
			A = res[param][0]
			R = res[param][1]
			F = res[param][2]
			print(f'({robustness_list.index(rtem)}, {F})',end=' ')
		print()

run('adult','race','FGSM')