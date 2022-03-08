import tqdm
import torch
import numpy as np
from TorchAttackable import TorchNeuralNetworks, TorchNNCore


class FaroInProc(TorchNeuralNetworks):
	def __init__(
		self, lr=0.01, n_epoch=500, method="FGSM", eps=0.1, hiddens=[], seed=None
	):
		super(FaroInProc, self).__init__(
			lr=lr, n_epoch=n_epoch, hiddens=hiddens, seed=seed
		)
		self._seed = seed
		if self._seed is not None:
			np.random.seed(self._seed)
			torch.manual_seed(self._seed)
		self._loss_func = torch.nn.BCELoss()
		self._method = method
		self._epsilon = eps
		# self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self._device = torch.device("cpu")

	def loss_fairness(self, y_pred, c):
		return torch.abs(torch.sum(y_pred * c))

	def loss_robustness_fgsm(self, paritial=True):
		noise = torch.sign(self._X.grad).detach()
		if not paritial:
			X_adv = (self._X + self._epsilon * noise).detach()
			y_adv_pred = self._model(X_adv)
			return self._loss_func(y_adv_pred, self._y)
		else:
			X_adv = (self._X[self._idx] + self._epsilon * noise[self._idx]).detach()
			y_adv_pred = self._model(X_adv)
			return self._loss_func(y_adv_pred, self._y[self._idx])

	def loss_robustness_pgd(self, paritial=True):
		X_adv = self._X.clone().detach().requires_grad_(True)
		for i in range(0,10):
			grad_X = torch.autograd.grad(self._loss_func(self._model(X_adv), self._y), X_adv)[0]
			noise = torch.sign(grad_X).detach()
			X_adv = (X_adv + self._epsilon*0.1*noise).detach().requires_grad_(True)
		X_adv.detach_()
		if not paritial:
			y_adv_pred = self._model(X_adv)
			return self._loss_func(y_adv_pred, self._y)
		else:
			y_adv_pred = self._model(X_adv[self._idx])
			return self._loss_func(y_adv_pred, self._y[self._idx])

	def fit(self, X, y, s=None, wR=0.0, wF=0.0, test=None, rep=False):

		report = {
			"epoch": [],
			"angle_ur": [],
			"angle_uf": [],
			"angle_rf": [],
			"loss_u": [],
			"loss_r": [],
			"loss_f": [],
			"train_metric": [],
			"train_metric_attack": [],
			"test_metric": [],
			"test_metric_attack": [],
		}

		self._idx = np.random.choice(
			np.arange(0, X.shape[0]), int(X.shape[0] * wR * 0.1), replace=False
		)

		self._X = torch.tensor(
			X, dtype=torch.float, requires_grad=True, device=self._device
		)
		self._y = torch.tensor(y.reshape(-1, 1), dtype=torch.float, device=self._device)
		if s is not None:
			self._s = torch.tensor(
				s.reshape(-1, 1), dtype=torch.float, device=self._device
			)
			self._c = (1 - self._s - self._s) / (
				(1 - self._s) * torch.sum(1 - self._s) + self._s * torch.sum(self._s)
			)

		self._model = TorchNNCore(
			inps=self._X.shape[1], hiddens=self._hiddens, seed=self._seed
		).to(self._device)

		optim = torch.optim.Adam(
			self._model.parameters(),
			lr=self._lr,  # weight_decay=wR
		)

		loss_r = torch.tensor(0.0, device=self._device)
		loss_f = torch.tensor(0.0, device=self._device)

		for epoch in tqdm.tqdm(range(self._n_epoch)):
			optim.zero_grad()

			y_pred = self._model(self._X)

			loss_u = self._loss_func(y_pred, self._y)

			if s is not None:
				loss_f = self.loss_fairness(y_pred, self._c)
			else:
				loss_f = None

			if self._X.grad is not None:
				if self._method == "FGSM":
					loss_r = self.loss_robustness_fgsm(paritial=True)
				elif self._method == "PGD":
					loss_r = self.loss_robustness_pgd(paritial=True)
				else:
					raise RuntimeError("Method must be FGSM or PGD.")
				self._X.grad = None
			else:
				loss_r = None

			loss = loss_u
			if wR > 0.0 and loss_r is not None:
				loss += wR * loss_r
			if wF > 0.0 and loss_f is not None:
				loss += wF * loss_f

			loss.backward()
			optim.step()

			if rep and (epoch % 5 == 1 or epoch == self._n_epoch - 1): # epoch == self._n_epoch - 1:

				# tmp_X_grad = self._X.grad
				# self._X.grad = None

				report["epoch"].append(epoch)
				report["loss_u"].append(loss_u.tolist() if loss_u is not None else None)
				report["loss_r"].append(loss_r.tolist() if loss_r is not None else None)
				report["loss_f"].append(loss_f.tolist() if loss_f is not None else None)
				report["train_metric"].append(self.metrics(X, y, s))
				report["train_metric_attack"].append(
					self.metrics_attack(X, y, s, method=self._method, use_y=True)
				)
				report["test_metric"].append(
					self.metrics(test["X"], test["y"], test["s"])
					if test is not None
					else None
				)
				report["test_metric_attack"].append(
					self.metrics_attack(
						test["X"],
						test["y"],
						test["s"],
						method=self._method,
						use_y=False,
					)
					if test is not None
					else None
				)

				# optim.zero_grad()
				# loss_u = self._loss_func(self._model(self._X), self._y)
				# loss_u.backward()
				# grad_u = [item.grad.tolist() for item in self._model.parameters()]

				# optim.zero_grad()
				# if self._method == "FGSM":
				# 	loss_r = self.loss_robustness_fgsm(paritial=False)
				# elif self._method == "PGD":
				# 	loss_r = self.loss_robustness_pgd(paritial=True)
				# else:
				# 	raise RuntimeError("Method must be FGSM or PGD.")
				# loss_r.backward()
				# grad_r = [item.grad.tolist() for item in self._model.parameters()]
				

				# optim.zero_grad()
				# loss_f = self.loss_fairness(self._model(self._X), self._c)
				# loss_f.backward()
				# grad_f = [item.grad.tolist() for item in self._model.parameters()]

				# report['angle_ur'].append(calc_angle(grad_u, grad_r))
				# report['angle_uf'].append(calc_angle(grad_u, grad_f))
				# report['angle_rf'].append(calc_angle(grad_r, grad_f))

				# self._X.grad = tmp_X_grad

		return report

def calc_angle(v1, v2, rad=False):
	u1 = np.array(flatten(v1))
	u2 = np.array(flatten(v2))
	u1 = u1 / np.linalg.norm(u1)
	u2 = u2 / np.linalg.norm(u2)
	dot = np.dot(u1, u2)
	angle = np.arccos(dot)
	if rad:
		return angle
	else:
		degree = (angle / np.pi) * 180
		return degree

def flatten(a):
	ret = []
	for item in a:
		if type(item) is list:
			ret.extend(flatten(item))
		else:
			ret.append(item)
	return ret

def experiments(data, attr, method="FGSM", wR=0.0, wF=0.0, seed=None, suffix=""):
	from utils import load_split
	import json, gzip

	train, test = load_split(data, attr)

	model = FaroInProc(lr=0.01, n_epoch=500, hiddens=[128], method=method, seed=seed)
	report = model.fit(
		train["X"], train["y"], s=train["s"], wR=wR, wF=wF, test=test, rep=True
	)
	report["data"] = data
	report["attr"] = attr
	report["method"] = method
	report["wR"] = wR
	report["wF"] = wF
	report["seed"] = seed

	print(report)

	f = gzip.open(f"./result/inproc/RnF_{suffix}.txt.gz", "at")
	f.write(json.dumps(report) + "\n")
	f.close()


if __name__ == "__main__":
	import sys, time

	if len(sys.argv) > 1:
		data = sys.argv[1]
		attr = sys.argv[2]
		method = sys.argv[3]
		wR = float(sys.argv[4])
		wF = float(sys.argv[5])
		suffix = sys.argv[6]
	else:
		data = "compas"
		attr = "race"
		method = "FGSM"
		wR = 0.0
		wF = 0.0
		suffix = ""

	seed = int(time.time())
	print("Seed is %d" % seed)
	print((data, attr, method, wR, wF))

	experiments(data, attr, method=method, wR=wR, wF=wF, seed=seed, suffix=suffix)
