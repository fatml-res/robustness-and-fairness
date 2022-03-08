import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_split(data,attr,del_s=True,ret_col=False):
	df=pd.read_csv(f'./data/{data}_train.csv')
	columns=list(df.columns)
	data_train=df.to_numpy()
	s_train=data_train[:,columns.index(attr)]
	c_train = (1 - s_train - s_train) / ((1 - s_train) * np.sum(1 - s_train) + s_train * np.sum(s_train))
	y_train=data_train[:,-1]
	X_train=np.delete(data_train[:,:-1],columns.index(attr),axis=1)
	if not del_s:
		X_train=np.hstack([s_train.reshape(-1,1),X_train])

	df=pd.read_csv(f'./data/{data}_test.csv')
	columns=list(df.columns)
	data_test=df.to_numpy()
	s_test=data_test[:,columns.index(attr)]
	y_test=data_test[:,-1]
	X_test=np.delete(data_test[:,:-1],columns.index(attr),axis=1)
	if not del_s:
		X_test=np.hstack([s_test.reshape(-1,1),X_test])

	train={'X':X_train,'s':s_train,'y':y_train, 'c':c_train}
	test={'X':X_test,'s':s_test,'y':y_test}

	if ret_col:
		columns.remove(attr)
		if not del_s:
			columns=[attr]+columns
		train['name']=columns
		test['name']=columns

	return train, test 

def calc_angle(v1,v2,rad=False):
	u1=v1/np.linalg.norm(v1)
	u2=v2/np.linalg.norm(v2)
	dot=np.dot(u1,u2)
	angle=np.arccos(dot)
	if rad:
		return angle
	else:
		degree=(angle/np.pi)*180
		return degree

def get_data(data,attr,binary=False):
	if binary:
		df=pd.read_csv(f'./data/{data}_binary.csv')
	else:
		df=pd.read_csv(f'./data/{data}.csv')
	metadata=df.to_numpy()
	metaattr=list(df.columns)

	metadata=np.array(metadata,dtype=float)

	if not binary:
		scaler=MinMaxScaler()
		metadata=scaler.fit_transform(metadata)

	s=metadata[:,metaattr.index(attr)].reshape(-1)
	y=metadata[:,-1].reshape(-1)
	X=metadata[:,:-1]
	X=np.delete(X,metaattr.index(attr),axis=1)
	X=np.hstack([s.reshape(-1,1),X])
	return X,y.reshape(-1,1)

if __name__=='__main__':
	print(calc_angle([-1,1],[0,10]))

