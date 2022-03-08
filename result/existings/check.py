import json

f=open('FnR.txt')
for row in f:
	row = json.loads(row)
	if row['data']=='hospital' and row['wR']==0.0 and row['wF']==0.0:
		print(row)
f.close()