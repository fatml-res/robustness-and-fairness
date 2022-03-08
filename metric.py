import numpy as np
from collections import OrderedDict


class Metric(object):
	def __init__(self, true, pred):

		self.true = np.array(true)
		if len(self.true.shape) == 2:
			if self.true.shape[1] == 1:
				self.true = self.true.reshape(-1)
			elif self.true.shape[1] == 2:
				self.true = self.true1[:, 1]
			else:
				raise ValueError("Only support binary classification")

		self.pred = np.array(pred)
		if len(self.pred.shape) == 2:
			if self.pred.shape[1] == 1:
				self.pred = self.pred.reshape(-1)
			elif self.pred.shape[1] == 2:
				self.pred = self.pred[:, 1]
			else:
				raise ValueError("Only support binary classification")

		self.threshold = 0.5

		self.cache = {}

	def set_threshold(threshold):
		self.threshold = threshold

	def tp(self):
		if "tp" in self.cache:
			return self.cache["tp"]
		ret = 0
		for i in range(0, len(self.true)):
			if self.true[i] > self.threshold and self.pred[i] > self.threshold:
				ret += 1
		self.cache["tp"] = ret
		return ret

	def tn(self):
		if "tn" in self.cache:
			return self.cache["tn"]
		ret = 0
		for i in range(0, len(self.true)):
			if self.true[i] < self.threshold and self.pred[i] < self.threshold:
				ret += 1
		self.cache["tn"] = ret
		return ret

	def fp(self):
		if "fp" in self.cache:
			return self.cache["fp"]
		ret = 0
		for i in range(0, len(self.true)):
			if self.true[i] < self.threshold and self.pred[i] > self.threshold:
				ret += 1
		self.cache["fp"] = ret
		return ret

	def fn(self):
		if "fn" in self.cache:
			return self.cache["fn"]
		ret = 0
		for i in range(0, len(self.true)):
			if self.true[i] > self.threshold and self.pred[i] < self.threshold:
				ret += 1
		self.cache["fn"] = ret
		return ret

	def precision(self):
		return self.tp() / (self.tp() + self.fp())

	def recall(self):
		return self.tp() / (self.tp() + self.fn())

	def accuracy(self):
		return (self.tp() + self.tn()) / (self.tp() + self.tn() + self.fp() + self.fn())

	def tpr(self):
		return self.tp() / (self.tp() + self.fn())

	def fpr(self):
		return self.fp() / (self.fp() + self.tn())

	def fscore(self):
		return 2.0/(1.0/self.precision()+1.0/self.recall())

	def report(self):
		ret = OrderedDict()
		ret["TP"] = self.tp()
		ret["TN"] = self.tn()
		ret["FP"] = self.fp()
		ret["FN"] = self.fn()
		ret["TPR"] = self.tpr()
		ret["FPR"] = self.fpr()
		ret["Precision"] = self.precision()
		ret["Recall"] = self.recall()
		ret["Accuracy"] = self.accuracy()
		return ret

	def varyth(self):
		ret = {}
		ret["Precision"] = []
		ret["Recall"] = []
		ret["TPR"] = []
		ret["FPR"] = []

		tmp_th = self.threshold

		for i in range(1, 100):
			self.set_threshold(i * 0.01)
			ret["Precision"].append(self.precision())
			ret["Recall"].append(self.recall())
			ret["TPR"].append(self.tpr())
			ret["FPR"].append(self.fpr())

		self.set_threshold(tmp_th)

		return ret

	def recall_disparity(self,s,absolute=True):
		z=np.array(s).astype(int)
		values=set(z)
		if len(values)<2:
			raise ValueError(f'Sensitive attribute only have {len(values)} value: {values}')
		if len(z.shape)==2:
			if z.shape[1]==1:
				z=z.reshape(-1)
			else:
				raise ValueError(f'Unexpected shape of sensitive attribute {z.shape}')
		ret={}
		for item in values:
			metric1=Metric(true=self.true[z==item],pred=self.pred[z==item])
			metric2=Metric(true=self.true[z!=item],pred=self.pred[z!=item])
			ret[item]=metric1.recall()-metric2.recall()
		if absolute:
			return abs(ret[0])
		else:
			return ret[1]

	def accuracy_disparity(self,s,absolute=True):
		z=np.array(s).astype(int)
		values=set(z)
		if len(values)<2:
			raise ValueError(f'Sensitive attribute only have {len(values)} value: {values}')
		if len(z.shape)==2:
			if z.shape[1]==1:
				z=z.reshape(-1)
			else:
				raise ValueError(f'Unexpected shape of sensitive attribute {z.shape}')
		ret={}
		for item in values:
			metric1=Metric(true=self.true[z==item],pred=self.pred[z==item])
			metric2=Metric(true=self.true[z!=item],pred=self.pred[z!=item])
			ret[item]=metric1.accuracy()-metric2.accuracy()
		if absolute:
			return abs(ret[0])
		else:
			return ret[1]

	def precision_disparity(self,s,absolute=True):
		z=np.array(s).astype(int)
		values=set(z)
		if len(values)<2:
			raise ValueError(f'Sensitive attribute only have {len(values)} value: {values}')
		if len(z.shape)==2:
			if z.shape[1]==1:
				z=z.reshape(-1)
			else:
				raise ValueError(f'Unexpected shape of sensitive attribute {z.shape}')
		ret={}
		for item in values:
			metric1=Metric(true=self.true[z==item],pred=self.pred[z==item])
			metric2=Metric(true=self.true[z!=item],pred=self.pred[z!=item])
			ret[item]=metric1.precision()-metric2.precision()
		if absolute:
			return abs(ret[0])
		else:
			return ret[1]

	def positive_disparity(self,s,absolute=True):
		value1=((self.pred>0.5)&(s==0)).sum()/(s==0).sum()
		value2=((self.pred>0.5)&(s==1)).sum()/(s==1).sum()
		if absolute:
			return abs(value1-value2)
		else:
			return value1-value2

	def truepos_disparity(self,s,absolute=True):
		return self.recall_disparity(s,absolute=absolute)
