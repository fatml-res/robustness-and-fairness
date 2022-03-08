import torch
import numpy as np

np.set_printoptions(suppress=False)
from torch.autograd import grad as gradient
from pandas import DataFrame
from matplotlib import pyplot as plt

from TorchAdversarial import TorchNNCore
from utils import load_split
from metric import Metric


class PreProcFlip(object):
	def __init__(
		self, data, attr, method="FGSM", k=10, seed=None, max_iter=1000, dR=0.1, dF=0.1
	):
		print((data, attr, method, dF, dR))

		self._data = data
		self._attr = attr
		self._seed = seed
		self._max_iter = max_iter
		self._epsilon = 0.1
		self._dR = dR
		self._dF = dF
		self._method = method
		if self._seed is not None:
			np.random.seed(self._seed)
			torch.manual_seed(self._seed)

		self._train_np, self._test_np = load_split(data, attr)

		if type(k) is int:
			self._k = k
		elif type(k) is float:
			self._k = np.ceil(self._train_np["X"].shape[0] * k).astype(int)
		else:
			raise ValueError("k must be float or int")

		print(
			"Flipping %.4f%% (%d) records every epoch"
			% (self._k * 100.0 / self._train_np["X"].shape[0], self._k)
		)

		self._train = {
			"X": torch.tensor(self._train_np["X"], dtype=torch.float),
			"y_orig": torch.tensor(
				self._train_np["y"].reshape(-1, 1), dtype=torch.float
			),
			"y": torch.tensor(self._train_np["y"].reshape(-1, 1), dtype=torch.float),
			"s": self._train_np["s"],
			"c": torch.tensor(self._train_np["c"].reshape(-1, 1), dtype=torch.float),
		}

		self._test = {
			"X": torch.tensor(self._test_np["X"], dtype=torch.float),
			"y": torch.tensor(self._test_np["y"].reshape(-1, 1), dtype=torch.float),
		}

	def _BCELoss(self, y_pred, y, reduction=True):
		if reduction:
			return -torch.mean(
				y * torch.log(0.99 * y_pred)
				+ (1.0 - y) * torch.log(1.0 - 0.99 * y_pred)
			)
		else:
			return -(
				y * torch.log(0.99 * y_pred)
				+ (1.0 - y) * torch.log(1.0 - 0.99 * y_pred)
			)

	def _DISPLoss(self, c, y_pred):
		return torch.square(torch.sum(c * y_pred))

	def _AccDisp(self, y, y_pred, s):
		metric = Metric(true=y.reshape(-1).tolist(), pred=y_pred.reshape(-1).tolist())
		return metric.accuracy(), metric.positive_disparity(s=s)

	def _Scaler(self, a_):
		a = np.array(a_)
		a_min = a.min()
		a_max = a.max()
		if a_min == a_max:
			return a
		else:
			return (a - a_min) / (a_max - a_min) * 2 - 1

	def _Hessian(self, model, X_train, y_train):
		y_pred = model(X_train)
		loss = self._BCELoss(y_pred, y_train)
		weights = model.layers[-2].weight
		bias = model.layers[-2].bias
		grad_L_w_1 = gradient(
			loss, (weights, bias), retain_graph=True, create_graph=True
		)
		Hessian = []
		for i in range(0, grad_L_w_1[0].shape[1]):
			grad_L_w_2 = gradient(
				grad_L_w_1[0][0][i], (weights, bias), retain_graph=True
			)
			Hessian.append(grad_L_w_2[0][0].tolist() + grad_L_w_2[1].tolist())
		grad_L_w_2 = gradient(grad_L_w_1[1][0], (weights, bias), retain_graph=True)
		Hessian.append(grad_L_w_2[0][0].tolist() + grad_L_w_2[1].tolist())
		return np.array(Hessian)

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

	def fit_transform(self, test_output=True, save=False):

		judgement_set = "train"

		res = {
			"setting": "%s_%s" % (self._data, self._attr),
			"train": {
				"orig": [],
				"attk": [],
				"disp": [],
			},
			"test": {
				"orig": [],
				"attk": [],
				"disp": [],
			},
			"iter": [],
			"valid": True,
		}

		model = TorchNNCore(
			inps=self._train["X"].shape[1],
			hiddens=[128],
			seed=self._seed,
			hidden_activation=torch.nn.LeakyReLU,
		)
		self._model = model

		optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
		loss_func = torch.nn.BCELoss()

		self._train["X_adv"] = None

		chosen_R = []
		chosen_F = []
		max_R_value = 0.0

		for it in range(0, self._max_iter):

			if self._train["X_adv"] is not None:
				X_train = torch.vstack(
					[self._train["X"], self._train["X_adv"]]
				).detach()
				y_train = torch.vstack(
					[self._train["y"], self._train["y"][chosen_R]]
				).detach()
			else:
				X_train = self._train["X"].detach()
				y_train = self._train["y"].detach()

			self._X_train = X_train
			self._y_train = y_train

			# BEGIN: Train modified model
			tolerence = 10
			last_loss = None
			for epoch in range(0, 1000):
				optim.zero_grad()
				y_pred = model(X_train)
				loss = self._BCELoss(y_pred, y_train)
				this_loss = loss.tolist()
				if last_loss is not None:
					if this_loss > last_loss or abs(last_loss - this_loss) < 1e-5:
						tolerence -= 1
					if tolerence == 0:
						break
				last_loss = this_loss
				loss.backward()
				optim.step()
			# END

			# Calculate inverse of Hessian Matrix
			Hessian_inv = np.linalg.inv(self._Hessian(model, X_train, y_train))
			# END

			# BEGIN: Common calculations
			weights = model.layers[-2].weight
			bias = model.layers[-2].bias
			y_pred = model(self._train["X"])
			orig_loss_utility = self._BCELoss(y_pred, self._train["y_orig"])
			orig_grad_utility = gradient(
				orig_loss_utility, (weights, bias), retain_graph=True
			)
			orig_grad_utility = np.array(
				orig_grad_utility[0][0].tolist() + orig_grad_utility[1].tolist()
			)
			# END: Common calculations

			# BEGIN: Pre-processing of fairness
			infl = -np.ones(self._train["s"].shape[0])
			if (
				len(res[judgement_set]["disp"]) > 0
				and res[judgement_set]["disp"][-1] > self._dF
				and len(chosen_F) < self._train["X"].shape[0]
			):
				y_pred = model(X_train)

				loss_orig = self._BCELoss(y_pred, y_train, reduction=False)
				loss_flip = self._BCELoss(y_pred, 1 - y_train, reduction=False)

				I_theta = []

				for i in range(0, self._train["y"].shape[0]):

					grad_loss_orig = gradient(
						loss_orig[i], (weights, bias), retain_graph=True
					)
					grad_loss_orig = np.array(
						grad_loss_orig[0][0].tolist() + grad_loss_orig[1].tolist()
					)

					grad_loss_flip = gradient(
						loss_flip[i], (weights, bias), retain_graph=True
					)
					grad_loss_flip = np.array(
						grad_loss_flip[0][0].tolist() + grad_loss_flip[1].tolist()
					)

					grad_diff = grad_loss_flip - grad_loss_orig

					I_theta.append(-np.dot(Hessian_inv, grad_diff).reshape(-1))

				y_pred = model(self._train["X"])
				orig_loss_fairness = self._DISPLoss(self._train["c"], y_pred)
				orig_grad_fairness = gradient(
					orig_loss_fairness, (weights, bias), retain_graph=True
				)
				orig_grad_fairness = np.array(
					orig_grad_fairness[0][0].tolist() + orig_grad_fairness[1].tolist()
				)

				for i in range(0, self._train["y"].shape[0]):
					infl[i] = np.dot(orig_grad_fairness, I_theta[i]) * np.exp(
						-abs(np.dot(orig_grad_utility, I_theta[i]))
					)

				indices = np.argsort(infl)
				chosen_F_set = set(chosen_F)
				actual_indices = []
				for item in indices:
					if infl[item] >= 0:
						break
					if item not in chosen_F_set:
						actual_indices.append(item)
						if len(actual_indices) == self._k:
							break
				chosen_F.extend(actual_indices)
				for item in actual_indices:
					self._train["y"][item][0] = 1.0 - self._train["y"][item][0]

			# END: Pre-processing of fairness

			# BEGIN: Pre-processing of robustness
			infl = -np.ones(self._train["s"].shape[0])
			if (
				len(res[judgement_set]["attk"]) > 0
				and res[judgement_set]["attk"][-1] < self._dR
				and len(chosen_R) < self._train["X"].shape[0]
			):

				X_adv = self.AdvExp(
					X=self._train["X"], y=self._train["y"], method=self._method
				)
				y_adv_pred = model(X_adv)
				loss_robustness = self._BCELoss(
					y_adv_pred, self._train["y"], reduction=False
				)

				I_theta = []

				for i in range(0, self._train["X"].shape[0]):
					grad_loss = gradient(
						loss_robustness[i], (weights, bias), retain_graph=True
					)
					grad_loss = np.array(
						grad_loss[0][0].tolist() + grad_loss[1].tolist()
					)
					I_theta.append(-np.dot(Hessian_inv, grad_loss).reshape(-1))

				y_pred_adv = model(X_adv)
				orig_loss_robustness = self._BCELoss(y_pred_adv, self._train["y_orig"])
				orig_grad_robustness = gradient(
					orig_loss_robustness, (weights, bias), retain_graph=True
				)
				orig_grad_robustness = np.array(
					orig_grad_robustness[0][0].tolist()
					+ orig_grad_robustness[1].tolist()
				)

				for i in range(0, self._train["y"].shape[0]):
					infl[i] = np.dot(orig_grad_robustness, I_theta[i]) * np.exp(
						-abs(np.dot(orig_grad_utility, I_theta[i]))
					)

				indices = np.argsort(infl)
				chosen_R_set = set(chosen_R)
				actual_indices = []
				for item in indices:
					if infl[item] >= 0:
						break
					if item not in chosen_R_set:
						actual_indices.append(item)
						if len(actual_indices) == self._k:
							break
				chosen_R.extend(actual_indices)
				if self._train["X_adv"] is None:
					self._train["X_adv"] = X_adv[actual_indices, :].detach()
				else:
					self._train["X_adv"] = torch.vstack(
						[
							self._train["X_adv"],
							X_adv[actual_indices, :].clone().detach(),
						]
					)
			# END: Pre-processing of robustness

			# BEGIN: Metrics of training
			y_pred = model(self._train["X"])
			metric = Metric(
				true=self._train_np["y"], pred=y_pred.detach().numpy().reshape(-1)
			)
			acc_train_org = metric.accuracy()
			disp_train_org = metric.positive_disparity(s=self._train_np["s"])
			y_pred_adv = model(
				self.AdvExp(X=self._train["X"], y=self._train["y"], method=self._method)
			)
			acc_train_atk = Metric(
				true=self._train_np["y"], pred=y_pred_adv.detach().numpy().reshape(-1)
			).accuracy()
			print(
				"Iter: %d, Train: (%.4f, %.4f, %.4f)"
				% (it + 1, acc_train_org, acc_train_atk, disp_train_org),
				end=", ",
			)
			res["train"]["orig"].append(acc_train_org)
			res["train"]["attk"].append(acc_train_atk)
			res["train"]["disp"].append(disp_train_org)
			# END: Metrics of training

			# BEGIN: Testing
			if test_output:
				# X_test.grad=None
				y_test_pred = model(self._test["X"])
				metric = Metric(
					true=self._test_np["y"],
					pred=y_test_pred.detach().numpy().reshape(-1),
				)
				acc_test_org = metric.accuracy()
				disp_test_org = metric.positive_disparity(s=self._test_np["s"])

				X_test_adv = self.AdvExp(X=self._test["X"], y=None, method=self._method)
				y_test_pred_atk = model(X_test_adv)
				acc_test_atk = Metric(
					true=self._test_np["y"],
					pred=y_test_pred_atk.detach().numpy().reshape(-1),
				).accuracy()

				print(
					"Test: (%.4f, %.4f, %.4f), Coverage: (R:%.2f, F:%.2f)"
					% (
						acc_test_org,
						acc_test_atk,
						disp_test_org,
						len(chosen_R) / self._train["X"].shape[0],
						len(chosen_F) / self._train["X"].shape[0],
					)
				)
				res["test"]["orig"].append(acc_test_org)
				res["test"]["attk"].append(acc_test_atk)
				res["test"]["disp"].append(disp_test_org)
				if res["test"]["attk"][-1] > max_R_value:
					max_R_value = res["test"]["attk"][-1]
			# END: Testing

			if (
				res[judgement_set]["attk"][-1] >= self._dR
				and res[judgement_set]["disp"][-1] <= self._dF
			):
				break

		if save:

			if self._train["X_adv"] is not None:
				X_train = torch.vstack(
					[self._train["X"], self._train["X_adv"]]
				).detach()
				y_train = torch.vstack(
					[self._train["y"], self._train["y"][chosen_R]]
				).detach()
			else:
				X_train = self._train["X"].detach()
				y_train = self._train["y"].detach()

			downstream_train = {
				"X": X_train.detach(),
				"y": y_train.detach(),
				"s": self._train["s"],
				"idx": chosen_R,
			}

			res["downstream_train"] = self._downstream_train
			res["downstream_test"] = self._test

		return res


def draw(res):

	fig, ax = plt.subplots(1, 2)
	fig.set_size_inches(12.8, 4.8)

	data = np.hstack(
		[
			np.array(res["train"]["orig"]).reshape(-1, 1),
			np.array(res["train"]["attk"]).reshape(-1, 1),
		]
	)
	df_train = DataFrame(
		data, index=res["iter"], columns=["Accuracy_Orig.", "Accuracy_Attk."]
	)
	main_ax = df_train.plot(ax=ax[0])
	main_ax.set_xlabel("Training epochs")
	main_ax.set_ylabel("Accuracy")
	main_ax.set_title(res["setting"] + "_train")

	data = np.array(res["train"]["disp"]).reshape(-1, 1)
	df_train = DataFrame(data, index=res["iter"], columns=["Disparity"])
	m2nd_ax = df_train.plot(secondary_y=True, ax=main_ax)
	m2nd_ax.set_ylabel("Statistical Parity")

	data = np.hstack(
		[
			np.array(res["test"]["orig"]).reshape(-1, 1),
			np.array(res["test"]["attk"]).reshape(-1, 1),
		]
	)
	df_test = DataFrame(
		data, index=res["iter"], columns=["Accuracy_Orig.", "Accuracy_Attk."]
	)
	main_ax = df_test.plot(ax=ax[1])
	main_ax.set_xlabel("Training epochs")
	main_ax.set_ylabel("Accuracy")
	main_ax.set_title(res["setting"] + "_test")

	data = np.array(res["test"]["disp"]).reshape(-1, 1)
	df_test = DataFrame(data, index=res["iter"], columns=["Disparity"])
	m2nd_ax = df_test.plot(secondary_y=True, ax=main_ax)
	m2nd_ax.set_ylabel("Statistical Parity")

	fig.tight_layout()


if __name__ == "__main__":

	import sys

	if len(sys.argv) >= 2:
		data = sys.argv[1]
		attr = sys.argv[2]
		method = sys.argv[3]
		dF = round(float(sys.argv[4]), 2)
		dR = round(float(sys.argv[5]), 2)
		k = 0.003
		if len(sys.argv > 6) and sys.argv[6].strip() == "save":
			saveflag = True
	else:
		data = "adult"
		attr = "race"
		method = "FGSM"
		dF = 0.6
		dR = 0.30
		k = 0.003
		saveflag = True

	import time

	seed = int(time.time())

	print("Seed is %d" % seed)

	model = PreProcFlip(
		data, attr, method=method, k=k, max_iter=1000, seed=seed, dR=dR, dF=dF
	)
	res = model.fit_transform(save=saveflag)

	res["data"] = data
	res["attr"] = attr
	res["method"] = method
	res["dF"] = dF
	res["dR"] = dR
	res["k"] = k
	res["seed"] = seed

	if saveflag:

		import pickle

		with open(
			"./result/predata/" + f"{data}_{attr}_{method}_{seed}.pre", "wb"
		) as handle:
			pickle.dump(res, handle)

	else:

		import json

		f = open(f"./result/preproc/FnR.txt", "a")
		f.write(json.dumps(res) + "\n")
		f.close()
