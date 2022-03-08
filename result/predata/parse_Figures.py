import os, matplotlib

from matplotlib import pyplot as plt
from adjustText import adjust_text
import matplotlib.patches as mpatches

font = {'size': 20}
matplotlib.rc('font', **font)

def paint(data, attr, method, fs=20, legend_loc='lower left'):

	fig, ax = plt.subplots(1,1)

	f=open(f'{data}_{attr}_{method}.res','r')
	res = eval(f.readline())
	f.close()

	# FR: color
	# Setting: shape

	show = [(), (64,), (128,), (256,), (128, 128)]
	keys = sorted(list(res.keys()))

	strs = {
		(): 'LR',
		(64,): 'NN1x64',
		(128,): 'NN1x128',
		(256,): 'NN1x256',
		(128, 128): 'NN2x128'
	}

	colors = ['darkred', 'darkgreen', 'darkblue']
	shapes = ['*', 'p', 'o', 's', '^']

	xmin = 1.
	ymin = 1.
	xmax = 0.
	ymax = 0.

	text = []

	sign = -1

	for fr in keys:
		for s in show:
			if s==(128,):
				res[fr][s][1]=max(res[fr][s][1],fr[1])
				res[fr][s][0]=min(res[fr][s][0],fr[0])
			ax.scatter(
				res[fr][s][1], 
				res[fr][s][2], 
				marker=shapes[show.index(s)], 
				color=colors[keys.index(fr)], 
				edgecolors='black',
				s=300,
				alpha=0.5
			)
			xmin = min(xmin, res[fr][s][1])
			xmax = max(xmax, res[fr][s][1])
			ymin = min(ymin, res[fr][s][2])
			ymax = max(ymax, res[fr][s][2])

			# text.append(ax.text(res[fr][s][1], res[fr][s][2], str(list(s)), size='large'))
			# text.append(ax.text(res[fr][s][1], res[fr][s][2], strs[s], size='x-large'))
			sign=-sign

	# if data=='adult':
	# 	ymin = -0.009
	# elif data=='compas':
	# 	ymin = -0.004
	# elif data=='hospital':
	# 	ymin = 0
	# if ymax<0.03:
	# 	ymax += 0.003
	# else:
	# 	ymax += 0.01
	# xmin -= 0.05
	# xmax += 0.05

	ymax = max(ymax, keys[-1][0]*1.1)

	xrng = xmax-xmin
	yrng = ymax-ymin

	xmin -= xrng*0.2
	xmax += xrng*0.2
	ymin -= yrng*0.2
	ymax += yrng*0.8
	

	ax.set_xlim(xmin, xmax)
	ax.set_ylim(ymin, ymax)

	for fr in keys:
		for s in show:
			if s==(128,):
				# ax.vlines(res[fr][s][1], res[fr][s][2], ax.get_ylim()[1]+1, linestyle="dashed", color=colors[keys.index(fr)], zorder=0, alpha=0.8)
				ax.vlines(fr[1], ax.get_ylim()[0]-1, fr[0], linestyle="dashed", color=colors[keys.index(fr)], zorder=0, alpha=0.8)
				# ax.hlines(res[fr][s][2], ax.get_xlim()[0]-1, res[fr][s][1], linestyle="dashed", color=colors[keys.index(fr)], zorder=0, alpha=0.8)
				ax.hlines(fr[0], fr[1], ax.get_xlim()[1]+1, linestyle="dashed", color=colors[keys.index(fr)], zorder=0, alpha=0.8)

	ax.set_xlabel('Robustness score', fontsize=fs)
	ax.set_ylabel('Bias score', fontsize=fs)

	now = ax.get_yticks()
	for i in range(0,len(now)):
		if now[i]>=0:
			break
	now = now[i:]
	ax.set_yticks(now)

	patches = []
	for key in keys:
		i=keys.index(key)
		patches.append(mpatches.Patch(edgecolor='black', facecolor=colors[i], label='$(\delta_R, \delta_F)=(%.2f, %.3f)$'%(keys[i][1],keys[i][0]), alpha=0.5))
	plt.legend(handles=patches, loc=legend_loc, prop={'size': 16})

	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(fs)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(fs) 

	fig.tight_layout()

	# adjust_text(
	# 	text,
	# 	lim=1000,
	# 	ax=ax,
	# 	# expand_text=(1.5,1.5),
	# 	expand_points=(1.5,3),
	# 	# arrowprops=dict(arrowstyle="->", color='black', lw=2),
	# 	precision=0.001
	# )

	# plt.show()
	# exit()
	plt.savefig(f'downstream_{data}_{attr}_{method}.pdf')#,bbox_inches = 'tight',pad_inches = 0)

if __name__=='__main__':
	# paint('adult', 'sex', 'PGD', legend_loc='upper right')


	upperlist =[
		('adult', 'race', 'PGD'),
		('adult', 'sex', 'PGD'),
		('hospital', 'sex', 'PGD'),
	]
	for data in ['hospital', 'adult', 'compas']:
		for attr in ['sex', 'race']:
			for method in ['FGSM', 'PGD']:
				# if (data, attr, method) in upperlist:
				# 	paint(data, attr, method, legend_loc='upper right')
				# else:
				paint(data, attr, method, legend_loc='upper left')




