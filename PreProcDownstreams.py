import os
import json
import gzip
import torch
import pickle
import numpy as np
np.set_printoptions(suppress=False)
from torch.autograd import grad as gradient
from pandas import DataFrame
from matplotlib import pyplot as plt

from TorchAdversarial import TorchNNCore
from TorchAdversarialTmp import TorchAdversarial
from utils import load_split
from metric import Metric
from copy import deepcopy


class Downstreams(object):
	def __init__(self, data, attr, method='FGSM'):

		self._data = data
		self._attr = attr
		self._method = method

		self._train_np, self._test_np = load_split(data, attr)
		self._test = {
			'X':torch.tensor(self._test_np['X'], dtype=torch.float),
			'y':torch.tensor(self._test_np['y'].reshape(-1,1), dtype=torch.float),
			's':self._test_np['s']
		}
		
		self._trains = {}
		self._result = {}
		self._models = {
			(), (64,), (128,), (256,), (128, 128)
		}
		
		fnames = os.listdir('./result/predata/')
		for fname in fnames:
			if '.pre' not in fname:
				continue
			setting = fname.split('_')
			if (setting[0], setting[1], setting[2]) != (data, attr, method):
				continue
			with open('./result/predata/'+fname, 'rb') as handle:
				res = pickle.load(handle)
				dR = res['dR']
				dF = res['dF']
				key = (dF, dR)
				if key not in self._trains:
					self._trains[key] = []
				self._trains[key].append(res['downstream_train'])
				self._trains[key][-1]['seed'] = res['seed']
				if self._trains[key][-1]['X'].shape[0] != self._trains[key][-1]['y'].shape[0]:
					l = min(self._trains[key][-1]['X'].shape[0], self._trains[key][-1]['y'].shape[0])
					self._trains[key][-1]['X'] = self._trains[key][-1]['X'][:l].detach()
					self._trains[key][-1]['y'] = self._trains[key][-1]['y'][:l].detach()
					self._trains[key][-1]['s'] = self._trains[key][-1]['s'][:l]
				if key not in self._result:
					self._result[key] = {}
					for param in self._models:
						self._result[key][param] = {'performance':np.zeros(3), 'count':0}

	def _BCELoss(self, y_pred, y, reduction=True):
		if y_pred.shape[1] != y.shape[1]:
			l = min([y_pred.shape[0], y.shape[1]])
			y_pred = y_pred[:l]
			y = y[:l]
		if reduction:			
			return -torch.mean(
				y * torch.log(0.99 * y_pred) + (1.0 - y) * torch.log(1.0 - 0.99 * y_pred)
			)
		else:
			return -(y * torch.log(0.99 * y_pred) + (1.0 - y) * torch.log(1.0 - 0.99 * y_pred))

	def AdvExp(self, X, y=None, method="FGSM", eps=0.1):
		if type(X) is torch.Tensor:
			X_ = X.clone().detach().requires_grad_(True)
		else:
			X_ = torch.tensor(X, dtype=torch.float, requires_grad=True)
		if method == "FGSM":
			y_pred = self._model(X_)
			if y is not None:
				y_ = torch.tensor(y.reshape(-1, 1), dtype=torch.float)
			else:
				y_ = torch.round(y_pred).detach()
			loss = self._BCELoss(y_pred, y_)
			noise = eps * torch.sign(torch.autograd.grad(loss, X_)[0])
			return (X_ + noise).detach()
		elif method == "PGD":
			if y is not None:
				y_ = torch.tensor(y.reshape(-1, 1), dtype=torch.float)
			else:
				y_ = None
			for i in range(0, 10):
				y_pred = self._model(X_)
				if y_ is None:
					y_ = torch.round(y_pred).detach()
				loss = self._BCELoss(y_pred, y_)
				noise = (eps * 0.1) * torch.sign(torch.autograd.grad(loss, X_)[0])
				X_ = (X_ + noise).detach().requires_grad_(True)
			return X_.detach()

	def run(self):

		for setting in self._trains: 
			for train in self._trains[setting]:
				for params in self._models:

					np.random.seed(train['seed'])
					torch.manual_seed(train['seed'])

					model = TorchNNCore(
						inps=train['X'].shape[1],
						hiddens=list(params),
						hidden_activation=torch.nn.ReLU,
					)
					self._model = model

					optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
					loss_func = torch.nn.BCELoss()

					tolerence = 10
					last_loss = None
					for epoch in range(0, 1000):
						optim.zero_grad()
						y_pred = model(train['X'])
						loss = self._BCELoss(y_pred, train['y'])
						this_loss = loss.tolist()
						if last_loss is not None:
							if this_loss > last_loss or abs(last_loss - this_loss) < 1e-5:
								tolerence -= 1
							if tolerence == 0:
								break
						last_loss = this_loss
						loss.backward()
						optim.step()

					y_test_pred = model(self._test['X'])
					metric = Metric(
						true=self._test['y'].reshape(-1).tolist(), pred=y_test_pred.detach().numpy().reshape(-1)
					)
					acc_test_org = metric.accuracy()
					disp_test_org = metric.positive_disparity(s=self._test['s'])

					X_test_adv = self.AdvExp(X=self._test['X'], y=None, method=self._method)
					y_test_pred_atk = model(X_test_adv)
					acc_test_atk = Metric(
						true=self._test['y'].reshape(-1).tolist(), pred=y_test_pred_atk.detach().numpy().reshape(-1)
					).accuracy()

					# print((self._data, self._attr, self._method), setting, params, np.round([acc_test_org, acc_test_atk, disp_test_org],4))

					self._result[setting][params]['performance'] += np.array([acc_test_org, acc_test_atk, disp_test_org])
					self._result[setting][params]['count'] +=1

		for setting in self._result:
			for params in self._result[setting]:
				self._result[setting][params] = (self._result[setting][params]['performance'] / self._result[setting][params]['count']).tolist()

		return self._result

	def distribution(self, ax, offset):
		tsize = self._train_np['X'].shape[0]
		ssize = np.array([(self._train_np['s']==0).sum(), (self._train_np['s']==1).sum()])
		keys = sorted(list(self._trains.keys()))

		for key in keys:
			newsize = np.zeros(2)
			count = 0
			for train in self._trains[key]:
				news = train['s'][tsize:]
				newsize += np.array([(news==0).sum(), (news==1).sum()])
				count+=1
			newsize = np.round(newsize/count).astype(int)
			ax.bar(offset+keys.index(key)*2, ssize[0], color='darkorange', edgecolor='black', alpha=0.9)
			ax.bar(offset+keys.index(key)*2+1, ssize[1], color='royalblue', edgecolor='black', alpha=0.9)
			ax.bar(offset+keys.index(key)*2, newsize[0], color='darkorange', edgecolor='black', alpha=0.5, bottom=ssize[0])
			ax.bar(offset+keys.index(key)*2+1, newsize[1], color='royalblue', edgecolor='black', alpha=0.5, bottom=ssize[1])
		


if __name__ == "__main__":

	action = 'run' # 'run' for running, 'draw' for drawing

	# if action == 'draw':

	# 	for data in ['adult', 'compas']:
	# 		for attr in ['race', 'sex']:
	# 			fig, ax = plt.subplots()
	# 			ax.set_title(f'{data}_{attr}')
	# 			for method in ['FGSM', 'PGD']:
	# 				downs = Downstreams(data, attr, method)
	# 				downs.distribution(ax, offset=0 if method=='FGSM' else 6+1)
				
	# 			ax.set_xticks(list(range(0,13)))
	# 			ax.set_xticklabels(
	# 				['S-Race','S-Race','M-Race','M-Race','W-Race','W-Race','','S-Gender','S-Gender','M-Gender','M-Gender','W-Gender','W-Gender'],
	# 				rotation = 45,
	# 				ha='right'
	# 			)
	# 			fig.tight_layout()
	# 			plt.show()
	# 			exit()

	if action =='run':
		for method in ['FGSM', 'PGD']:
			for data in ['adult','compas','hospital']:
				for attr in ['race', 'sex']:ø
					print(data, attr, method)
					downs = Downstreams(data, attr, method)
					result = downs.run()
					f=open(f'./result/predata/{data}_{attr}_{method}.res','w')
					f.write(str(result))
					f.close()

