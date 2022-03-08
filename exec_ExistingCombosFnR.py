import os, time, json

turn = 30
wFs={
	'fairness_reweighing':[0.0,1.0],
	'fairness_disparate':[0.0,0.2,0.4,0.6,0.8,1.0],
}
wRs=[0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
data_list = ["hospital","compas", "adult"]
attr_list = ["race", "sex"]
method_list = ['FGSM', 'PGD']

count=0
runtime=0.0
allct=0

executed={}

for it in range(1,turn+1):
	for data in data_list:
		for attr in attr_list:
			for method in method_list:
				for func in ['fairness_reweighing','fairness_disparate']:#,'fairness_adversarial']:
					for wF in wFs[func]:
						for wR in wRs:
							combo = (data, attr, method, func, wF, wR)
							if combo not in executed:
								executed[combo]=0
							allct+=1

if os.path.exists('./result/existings/FnR.txt'):
	f=open('./result/existings/FnR.txt')
	for line in f:
		obj=json.loads(line)
		combo = (obj['data'], obj['attr'], obj['method'], obj['func'], obj['wF'], obj['wR'])
		if combo in executed:
			executed[combo]+=1
			allct-=1

for it in range(1,turn+1):
	try:
		for data in data_list:
			for attr in attr_list:
				for method in method_list:
					for func in ['fairness_reweighing','fairness_disparate']:#,'fairness_adversarial']:
						for wF in wFs[func]:
							for wR in wRs:

								combo = (data, attr, method, func, wF, wR)

								if combo in executed and executed[combo]>0:
									executed[combo]-=1
									print('Combo %s is already calculated, skip.'%str(combo))
									continue

								start_t=time.time()

								os.system('python ExistingCombosFnR.py %s %s %s %s %.2f %.2f'%(data, attr, method, func, wF, wR))
								count+=1
								time.sleep(1)

								end_t=time.time()

								runtime+=end_t-start_t

								print('Finished %d / %d'%(count, allct))
								print('Runtime: %.2f seconds, Avg.: %.2f, Est.: %.2f'%(end_t-start_t, runtime/count, (allct-count)*runtime/count))
								print('')
						
	except KeyboardInterrupt:
		break