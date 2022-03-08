import matplotlib, gzip, json, os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import interpolate
import seaborn as sns

font = {'size': 22}

matplotlib.rc('font', **font)

def draw(data, attr, method):

	plt.clf()

	if not os.path.exists(f'./heatmap_preproc_{data}_{attr}_{method}.txt') or not os.path.exists(f'./heatmap_inproc_{data}_{attr}_{method}.txt'):
		print(f'{data}_{attr}_{method}: data incomplete.')
		return

	f=open(f'heatmap_preproc_{data}_{attr}_{method}.txt')
	pre = json.loads(f.readline())
	f.close()

	f=open(f'heatmap_inproc_{data}_{attr}_{method}.txt')
	inp = json.loads(f.readline())
	f.close()

	fmin = min(min(pre['F']), min(inp['F']))
	fmax = max(max(pre['F']),max(inp['F']))
	rmin = min(min(pre['R']), min(inp['R']))
	rmax = max(max(pre['R']),max(inp['R']))

	ff_ = np.linspace(fmin,fmax,num=101)
	rr_ = np.linspace(rmin,rmax,num=101)
	rr, ff = np.meshgrid(rr_,ff_)

	pre['grid'] = interpolate.griddata((pre['R'], pre['F']), pre['A'], (rr.ravel(), ff.ravel()))
	inp['grid'] = interpolate.griddata((inp['R'], inp['F']), inp['A'], (rr.ravel(), ff.ravel()))

	pre['np'] = pd.DataFrame(data={'x':rr.ravel(), 'y':ff.ravel(), 'z':pre['grid']}).pivot(index='y', columns='x', values='z').to_numpy()
	inp['np'] = pd.DataFrame(data={'x':rr.ravel(), 'y':ff.ravel(), 'z':inp['grid']}).pivot(index='y', columns='x', values='z').to_numpy()

	diff = pre['np'] - inp['np']
	only = np.zeros(diff.shape)
	for i in range(diff.shape[0]):
		for j in range(diff.shape[1]):
			diff[i][j]=np.sign(diff[i][j])*0.4
			if np.isnan(pre['np'][i][j]) and not np.isnan(inp['np'][i][j]):
				only[i][j]=-1.0
			elif not np.isnan(pre['np'][i][j]) and np.isnan(inp['np'][i][j]):
				only[i][j]=1.0
			else:
				only[i][j]=np.nan
	
	# only*=0.9
	# diff*=

	# fig, ax = plt.subplots()
	plt.pcolormesh(rr_.ravel(), ff_.ravel(), diff, cmap='PRGn', vmin=-1, vmax=1)
	plt.pcolormesh(rr_.ravel(), ff_.ravel(), only, cmap='coolwarm', vmin=-1, vmax=1)
	# cbar = plt.colorbar(heatmap)
	xtick_min = np.ceil(rmin*100)/100
	xtick_max = np.floor(rmax*100)/100
	xtick_bins = 6
	xtick_step = (xtick_max - xtick_min) / (xtick_bins - 1)
	xticks = np.round(np.arange(xtick_min, xtick_max, xtick_step), 2).tolist()
	if xtick_max - xticks[-1] > 0.75*xtick_step:
		xticks.append(xtick_max)
	print(xticks)
	plt.xticks(xticks)

	plt.xlabel('Robustness threshold')
	plt.ylabel('Fairnesss threshold')
	plt.tight_layout()
	# plt.show()
	plt.savefig(f'diffmap_{data}_{attr}_{method}.pdf')


if __name__=='__main__':
	for data in ['adult','compas','hospital']:
		for attr in ['race', 'sex']:
			for method in ['FGSM', 'PGD']:
				try:
					draw(data, attr, method)
				except:
					pass