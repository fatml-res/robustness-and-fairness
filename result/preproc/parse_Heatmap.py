import matplotlib, gzip, json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import interpolate
import seaborn as sns

font = {'size': 22}

matplotlib.rc('font', **font)

minmax={}

def collect_colorbar_range(data):
	if data in minmax:
		return minmax[data][0], minmax[data][1]
	Accs = []
	f=open('./FnR.txt','r')
	for row in f:
		obj=json.loads(row)
		if obj['data'] != data:
			continue
		if not obj['valid']:
			continue
		acc = obj['test']['orig'][-1]
		atk = obj['test']['attk'][-1]
		dsp = obj['test']['disp'][-1]
		dR = obj['dR']
		dF = obj['dF']
		if dsp>dF or atk<dR:
			continue
		Accs.append(acc)
	minmax[data] = (min(Accs), max(Accs))
	return min(Accs), max(Accs)

def draw(data, attr, method):

	print(data, attr, method)

	vmin = None
	vmax = None
	# vmin, vmax = collect_colorbar_range(data)

	plt.clf()
	res = {}
	f=open('./FnR.txt','r')
	for row in f:
		obj=json.loads(row)
		if (obj['data'], obj['attr'], obj['method']) != (data, attr, method):
			continue
		dR = obj['dR']
		dF = obj['dF']
		param = (dR, dF)
		if param not in res:
			res[param]=np.zeros(3)
		acc = obj['test']['orig'][-1]
		atk = obj['test']['attk'][-1]
		dsp = obj['test']['disp'][-1]
		if dsp>dF or atk<dR:
			continue
		if not obj['valid']:
			continue
		res[param]=np.max([[acc, atk, dsp],res[param].tolist()],axis=0)
	f.close()

	if len(res)!=400:
		print(f'{data}_{attr}_{method} Data incomplete, nothing was done.')
		return
		
	Fs = []
	Rs = []
	As = []
	for param in res:
		if res[param].tolist()==[0,0,0]:
			continue
		As.append(res[param][0])
		Rs.append(res[param][1])
		Fs.append(res[param][2])

	mapdata = {
		'A':As,
		'R':Rs,
		'F':Fs,
	}
	f=open(f'../preindiff/heatmap_preproc_{data}_{attr}_{method}.txt','w')
	f.write(json.dumps(mapdata))
	f.close()

	# ff = np.linspace(np.min(Fs), np.max(Fs))
	# rr = np.linspace(np.min(Rs), np.max(Rs))
	fmin=np.floor(np.min(Fs)*100)/100
	fmax=np.ceil(np.max(Fs)*100)/100
	# rmin=np.floor(np.min(Rs)*10)/10
	# rmax=np.ceil(np.max(Rs)*10)/10
	print(fmin, fmax)
	print(np.min(Rs), np.max(Rs))
	print(np.min(Fs), np.max(Fs))
	ff_ = np.linspace(np.min(Fs),np.max(Fs),num=101)
	rr_ = np.linspace(np.min(Rs),np.max(Rs),num=101)
	rr, ff = np.meshgrid(rr_, ff_)
	print(rr)
	aa = interpolate.griddata((Rs, Fs), As, (rr.ravel(), ff.ravel()))
	# xx.ravel(), yy.ravel()
	
	dataset = pd.DataFrame(data={'x':rr.ravel(), 'y':ff.ravel(), 'z':aa})
	dataset = dataset.pivot(index='y', columns='x', values='z')
	print(dataset.to_numpy().shape)
	# heatmap=plt.pcolor(rr_.ravel(), ff_.ravel(), dataset.to_numpy())
	# plt.colorbar(heatmap)

	fig, ax = plt.subplots()
	if vmin==vmax==None:
		heatmap = ax.pcolor(rr_.ravel(), ff_.ravel(), dataset.to_numpy())
	else:
		heatmap = ax.pcolor(rr_.ravel(), ff_.ravel(), dataset.to_numpy(), vmin=vmin, vmax=vmax)
	cbar = plt.colorbar(heatmap)
	cbar.set_label('Accuracy')
	xtick_min = np.floor(np.min(Rs)*100)/100
	xtick_max = np.ceil(np.max(Rs)*100)/100
	xtick_bins = 5
	xtick_step = (xtick_max - xtick_min) / (xtick_bins - 1)
	xticks = np.round(np.arange(xtick_min, xtick_max, xtick_step), 2).tolist()
	if xtick_max - xticks[-1] > 0.75*xtick_step:
		xticks.append(xtick_max)
	print(xticks)
	ax.set_xticks(xticks)

	ztick_min = np.ceil(np.min(As)*1000)/1000
	ztick_max = np.floor(np.max(As)*1000)/1000
	ztick_bins = 5
	ztick_step = (ztick_max - ztick_min) / (ztick_bins - 1)
	zticks = np.round(np.arange(ztick_min, ztick_max, ztick_step), 3).tolist()
	if ztick_max - zticks[-1] > 0.75*ztick_step:
		zticks.append(ztick_max)
	print(zticks)
	cbar.set_ticks(zticks)

	ax.ticklabel_format(style='sci', scilimits=(-2,2), axis='y')

	plt.xlabel('Robustness score')
	plt.ylabel('Bias score')
	plt.tight_layout()
	plt.savefig(f'heatmap_preproc_{data}_{attr}_{method}.pdf')
	
	res={
		'data':data,
		'attr':attr,
		'method':'FGSM',
		'xticks':np.arange(np.ceil(np.min(Rs)*10)/10,np.floor(np.max(Rs)*10)/10+0.1,0.1).tolist(),
		'R':rr_.ravel().tolist(),
		'F':ff_.ravel().tolist(),
		'A':dataset.to_numpy().tolist()
	}
	

if __name__=='__main__':
	# draw('compas', 'race', 'FGSM')
	# exit()
	for data in ['adult','compas','hospital']:
		for attr in ['race', 'sex']:
			for method in ['FGSM', 'PGD']:
				draw(data, attr, method)