import os, time, json, gzip

wR_list = [
	0.00,
	0.01,
	0.02,
	0.03,
	0.04,
	0.05,
	0.06,
	0.07,
	0.08,
	0.09,
	0.10,
	0.20,
	0.30,
	0.40,
	0.50,
	0.60,
	0.70,
	0.80,
	0.90,
	1.00,
]
wF_list = [round(0.05 * i, 2) for i in range(0, 21)]

turn = 10
data_list = ["compas", "adult", "hospital"]
attr_list = ["race", "sex"]
method_list = ['FGSM','PGD']

count = 0
allct = 0
runtime = 0.0

executed = {}

for it in range(1, turn+1):
	for data in data_list:
		for attr in attr_list:
			for method in method_list:
				for wR in wR_list:
					for wF in wF_list:
						combo = (data, attr, method, wR, wF)
						if combo not in executed:
							executed[combo]=0
						allct+=1

for it in range(1, turn+1):
	if os.path.exists("./result/inproc/RnF_%d.txt.gz" % it):
		f = gzip.open("./result/inproc/RnF_%d.txt.gz" % it)
		for line in f:
			obj = json.loads(line)
			combo = (obj["data"], obj["attr"], obj["method"], obj["wR"], obj["wF"])
			if combo in executed:
				executed[combo]+=1
				allct-=1
		f.close()

print(f'There are {allct} tasks in total.')

for it in range(1, 11):
	try:
		for data in data_list:
			for attr in attr_list:
				for method in method_list:
					for wR in wR_list:
						for wF in wF_list:
							combo = (data, attr, method, wR, wF)
							if combo in executed and executed[combo] > 0:
								executed[combo] -= 1
								print("Combo", combo, "is already calculated, skipped.")
								continue

							start_t = time.time()

							os.system(
								"python InProcess.py %s %s %s %.2f %.2f %s"
								% (data, attr, method, wR, wF, "%d" % it)
							)
							count += 1
							time.sleep(1)

							end_t = time.time()

							runtime += end_t - start_t

							print("Finished %d / %d" % (count, allct))
							print(
								"Runtime: %.2f seconds, Avg.: %.2f, Est.: %.2f"
								% (
									end_t - start_t,
									runtime / count,
									(allct - count) * runtime / count,
								)
							)
							print("")

	except KeyboardInterrupt:
		break
