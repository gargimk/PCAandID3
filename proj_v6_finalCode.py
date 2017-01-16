import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as PCA
import math
from sklearn import decomposition
from sklearn import datasets
from scipy import linalg
from scipy.sparse import issparse, csr_matrix
import pandas as pd

def read_data():
	df = pd.read_csv(
		filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
		header=None,
		sep=',')

	df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
	df.dropna(how="all", inplace=True) # drops the empty line at file-end
	df.tail()
	
	return df

def algoPCA(df, component):   
	
	data = df.ix[:,0:4].values  #feature matrix
	targetClass= df.ix[:,4].values #class matrix

	X = StandardScaler().fit_transform(data)  #standarize data
	mean_vec = np.mean(X, axis=0)
	X = X-mean_vec
	
	cov_mat = np.cov(X.T)  #calculate the covariance matrix
	print('Covariance Matrix:\n', cov_mat)

	eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)   # calculate eigen values and vectors
	print('Eigenvectors \n%s' %eig_vec_cov)
	print('\nEigenvalues \n%s' %eig_val_cov)
	
	#calculate the explained variance
	tot = sum(eig_val_cov)
	exp_var_ratio = [(i / tot) for i in sorted(eig_val_cov, reverse=True)]
	print('\nExplained variance ratio(user defined function) \n%s' %exp_var_ratio[:component])
	
	# Make a list of (eigenvalue, eigenvector) tuples
	eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]

	# Sort the (eigenvalue, eigenvector) tuples from high to low
	eig_pairs.sort()
	eig_pairs.reverse()
	print('\nEigen Pairs \n%s' %eig_pairs)
	
	#calculate matrix_W
	temp = eig_pairs[0][1].reshape(4,1)
	for n in range(1,component):
		temp = np.hstack((temp,eig_pairs[n][1].reshape(4,1)))
		matrix_w = temp
	print("Matrix W:\n", matrix_w)

	pcaOutput = X.dot(matrix_w)  # calculate the principal components
	
	pcaOutput[:, 1] = -pcaOutput[:, 1]  # work around to match the values with inbuilt function
		
	return pcaOutput;

def inbuiltPCA(df, component):
	data = df.ix[:,0:4].values
	X_std = StandardScaler().fit_transform(data)
	sklearn_pca = PCA(n_components=component)
	inbuiltPCAOutput = sklearn_pca.fit_transform(X_std)
	print('\nExplained variance ratio(inbuilt function) \n%s' %sklearn_pca.explained_variance_ratio_) 
	return inbuiltPCAOutput
	
def plot2PC(PCAOutput,y,msg,i):

	with plt.style.context('seaborn-whitegrid'):
		plt.figure(i,figsize=(6, 4))
		for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
							('blue', 'red', 'green')):
			plt.scatter(PCAOutput[y==lab, 0],
						PCAOutput[y==lab, 1],
						label=lab,
						c=col)
		plt.title(msg)
		plt.xlabel('Principal Component 1')
		plt.ylabel('Principal Component 2')
		plt.legend(loc='lower center')
		plt.tight_layout()
		plt.show()
		
def plot3PC(PCAOutput,y,msg,i):
	centers = [[1, 1], [-1, -1], [1, -1]]
	fig = plt.figure(i, figsize=(8, 8))
	plt.clf()
	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
	plt.cla()
	
	for index, item in enumerate(y):
		if (item == 'Iris-setosa'):
			y[index] = 0
		if(item == 'Iris-versicolor'):
			y[index] = 1
		if(item == 'Iris-virginica'):
			y[index] = 2
	y=y.astype(np.int32)
	for name, label in [('setosa', 0), ('versicolor', 1), ('virginica', 2)]:
		ax.text3D(PCAOutput[y == label, 0].mean(),
				  PCAOutput[y == label, 1].mean() + 1.5,
				  PCAOutput[y == label, 2].mean(), name,
				  horizontalalignment='center',
				  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
				  

	# Reorder the labels to have colors matching the cluster results
	y = np.choose(y, [1, 2, 0]).astype(np.float)
	
	ax.scatter(PCAOutput[:, 0], PCAOutput[:, 1], PCAOutput[:, 2], c=y, cmap=plt.cm.spectral)
	ax.w_xaxis.set_ticklabels([])
	ax.w_yaxis.set_ticklabels([])
	ax.w_zaxis.set_ticklabels([])
	ax.set_title(msg)
	ax.set_xlabel('Principal Component 1')
	ax.set_ylabel('Principal Component 2')
	ax.set_zlabel('Principal Component 3')
	plt.show()

# read data
df_implemented = read_data()
y= df_implemented.ix[:,4].values
# user defined PCA with PC=2	
Y_userDef_PC2 =	algoPCA(df_implemented,2)
#plot the 2D graph
plot2PC(Y_userDef_PC2,y,'Results of user defined function for PC=2',1)
# user defined PCA with PC=3
Y_userDef_PC3 =	algoPCA(df_implemented,3)
#plot the 3D graph
plot3PC(Y_userDef_PC3,y,'Results of user defined function for PC=3',2)

# read data
df_inbuilt = read_data()
y=df_inbuilt.ix[:,4].values
# inbuilt PCA for PC=2
Y_sklearn_PC2=inbuiltPCA(df_inbuilt,2)
#plot the 2D graph
plot2PC(Y_sklearn_PC2,y,'Results of inbuilt function for PC=2',3)
# inbuilt PCA for PC=3
Y_sklearn_PC3=inbuiltPCA(df_inbuilt,3)
#plot the 3D graph
plot3PC(Y_sklearn_PC3,y,'Results of inbuilt function for PC=3',4)

