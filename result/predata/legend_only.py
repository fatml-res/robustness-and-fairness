import os, matplotlib

from matplotlib import pyplot as plt
from adjustText import adjust_text
import matplotlib.patches as mpatches

from numpy.random import randn

z = randn(10)

keys=[(0.01, 0.65), (0.015, 0.55), (0.02, 0.45)]

colors = ['darkred', 'darkorange', 'darkblue']
shapes = ['*', 'p', 'o', 's', '^']
models = ['LR','NN1x64','NN1x128','NN1x256','NN2x128']

handles = []
for i in range(0,5):
	plt.scatter([1],[1], marker=shapes[i], s=100, edgecolor='k', facecolor='w')
plt.legend(
	models,
	ncol=5
)
plt.show()