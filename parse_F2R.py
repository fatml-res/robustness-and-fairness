import json
import numpy as np

data = {}

f=open('./result/existings/F2R.txt')
for line in f:
	obj=json.loads(line)
	combo=(obj['method'],obj['func'],obj['data'],obj['attr'],obj['param'])
	if combo not in data:
		data[combo]={
			'orig':0.0,
			'fair':0.0,
			'count':0
		}
	data[combo]['orig']+=obj['result_orig']['test_adv'][0]
	data[combo]['fair']+=obj['result_fair']['test_adv'][0]
	data[combo]['count']+=1
f.close()

for combo in data:
	data[combo]['orig']/=data[combo]['count']
	data[combo]['fair']/=data[combo]['count']

for combo in data:
	data[combo]['score_orig']=1-data[combo]['orig']
	data[combo]['score_fair']=1-data[combo]['fair']
	data[combo]['score']=data[combo]['score_fair']-data[combo]['score_orig']
	# print(combo, data[combo]['score_fair'], data[combo]['score_orig'])

i=0
for combo in sorted(data.keys()):
	print(combo, round(data[combo]['score'],4))
