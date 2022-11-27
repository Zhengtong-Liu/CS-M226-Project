import tensorflow as tf
import numpy as np
import pandas as pd
from utils import *
from customModel import *
import matplotlib.pyplot as plt
import itertools

class RFF_gpytorch:

    def __init__(self, gamma, n_components,seed=None):
        assert n_components%2 ==0
        self.gamma = gamma
        self.n_components = n_components
        self.seed = seed

    def fit(self,X):
        seed = self.seed
        D = self.n_components
        M = X.shape[1]
        self.X = X
        self.M = M
        if seed is not None:
            np.random.seed(seed)
        W = np.random.standard_normal((M,D//2))
        self.W = W
        self.seed = seed
    
    def transform(self, X, active_dims=None):
        if active_dims is None:

            active_dims = np.array([True for i in range(X.shape[1])])
        
        W = self.W[active_dims,0:int(self.n_components//2 * np.mean(active_dims))]
        X = X[:,active_dims]
        D = int(self.n_components*np.mean(active_dims))
        gamma = self.gamma
        W = np.sqrt(2*gamma)*W
        projection = X@W
        sin = np.sin(projection)
        cos = np.cos(projection)
        Combine = np.empty((sin.shape[0],2*sin.shape[1]), dtype=float)
        Combine[:,0::2] = sin
        Combine[:,1::2] = cos
        Combine *= np.sqrt(2.)/np.sqrt(D)

        return np.float32(Combine)
        
    def fit_transform(self,X,active_dims=None):
        if active_dims is None:
            active_dims = np.array([True for i in range(X.shape[1])])
        

        seed = self.seed
        X = X[:,active_dims]
        self.X = X
        gamma = self.gamma
        D = int(self.n_components//2*np.mean(active_dims))
		
        if seed is not None:
            np.random.seed(seed)
        M = X.shape[1]
        W = np.random.standard_normal((M,D//2))*np.sqrt(2*gamma)
        self.W = W
        projection = X@W
        sin = np.sin(projection)
        cos = np.cos(projection)
        Combine = np.empty((sin.shape[0],2*sin.shape[1]), dtype=float)
        Combine[:,0::2] = sin
        Combine[:,1::2] = cos
        Combine *= np.sqrt(2.)/np.sqrt(D)
        return np.float32(Combine)


class BANNs(object):
	def __init__(self,X, y, mask, centered=False, maf=None, nModelsSNP=30, nModelsSET=30, nModelsSET_nonlinear=30, epi_p=None, max_epi=100, automated=False):
		print("Welcome to BANNs. Please make sure SNPs in the SNP List you provide are in the same order as in the genotype matrix. Results we return will be in the order of SNP annotations and SNP-set annotations.")
		self.X=X
		self.y=y
		self.mask=mask
		self.nModelsSNP=nModelsSNP
		self.nModelsSET=nModelsSET
		self.nModelsSET_nonlinear = nModelsSET_nonlinear
		# self.n_components = n_components
		self.epi_p = epi_p
		self.max_epi = max_epi
		self.optimizer=tf.compat.v1.train.GradientDescentOptimizer(1e-4, use_locking=False, name='GradientDescent')
		self.checkInputs()

		if maf!=None:
			self.QC_SNPs()

		if centered==False:
			self.center_scale_inputs()

	def checkInputs(self):
		try:
			self.X=np.asarray(self.X)
			self.y=np.asarray(self.y)
			self.mask=np.asarray(self.mask)
		except:
			print("Please make sure to give numerical matrices and vectors for X, y, and annotation mask")

		if(np.isnan(self.X).any()):
			print("X genotype matrix contains NaN values. Please input a matrix with no NaN values")
			return
		if (np.isnan(self.y).any()):
			print("y phenotype vector contains NaN values. Please input a vector with no NaN values")
			return
		if (np.isnan(self.mask).any()):
			print("SNP-SNPset mask matrix contains NaN values. Please input a matrix with no NaN values")
			return
		if((isinstance(self.nModelsSNP, int) and (isinstance(self.nModelsSET,int)) and (isinstance(self.nModelsSET_nonlinear,int)))==False):
			print("nModelsSNP, nModelsSET and nModelsSET_nonlinear parameters, which determine the number of models to initialize for SNP and SNP-Set and SNP-set(nonlinear) layers, respectively, should be integers")
			return 
		if self.epi_p != None:
			if (isinstance(self.epi_p, float) == False) or (self.epi_p >= 1) or (self.epi_p <= 0):
				print("Invalid epi_p input, ignoring this parameter")
				self.epi_p = None
		if(isinstance(self.max_epi, int) == False) or (self.max_epi <= 0):
			print("max_epi, the maximum number of pairwise interactions in a SNP-set, should be an nonnegative integer")
			return 
			
		#### Get input shapes:
		#Number of individuals (or SNPs if using summary statistics) from genotype matrix:
		N=self.X.shape[0]
		#Number of SNPs:
		p=self.X.shape[1]
		#Number of individuals from phenotype array:
		Ny=self.y.shape[0]
		#Number of SNPs from mask files:
		pm=self.mask.shape[0]
		#Number of genes from mask file:
		g=self.mask.shape[1]

		### Check if shapes agree:
		if(N!=Ny):
			print("Number of samples do not match in X matrix and y vector")
			return
		if(p!=pm):
			print("Number of SNPs do not match in X matrix (number of columns) and annotation mask matrix (number of rows)")
			return

	def QC_SNPs(self, maf):
		currentMAF=np.mean(self.X, axis=0)
		self.X=self.X[:,currentMAF>maf]

	def center_scale_inputs(self):
		self.X=np.nan_to_num((self.X-np.mean(self.X, axis=0))/np.std(self.X,axis=0)) # Standardized genotype matrix based on means and standard deviations
		self.y=np.nan_to_num((self.y-np.mean(self.y))/np.std(self.y)) # Standardized phenotype array based on means and standard deviations

	def estimatePVE(self,layer,X):
		p=X.shape[1]
		pve=np.repeat(0.0,100)
		for i in range(0,100):
			j = np.random.choice(layer.models,1,p=layer.w)
			b=layer.kernel[:,j]+np.sqrt(layer.s[:,j])*np.random.normal(0,1,p)
			b=b*(np.random.uniform(0,1,p)<layer.pip[:,j])
			sz=var1(np.matmul(X, b))
			pve[i]=sz/(sz+layer.tau[j])
		return np.mean(pve)

	def run(self, nonlinear_flag=False):
		self.SNP_layer=HiddenLayer(self.X, self.y, self.nModelsSNP)
		self.SNP_layer.train(self.X, self.y, 10000)
		self.SNP_layer.w = normalizelogweights(self.SNP_layer.logw)
		b=rep_col(np.sum(self.SNP_layer.w * self.SNP_layer.pip*self.SNP_layer.kernel, axis=1),self.mask.shape[1]) ## beta here
		self.G=np.matmul(self.X,self.mask*b)
		print(self.G.shape)
		self.SET_layer=HiddenLayer(self.G,self.y,self.nModelsSET)
		self.SET_layer.train(self.G, self.y, 10000)
		self.SET_layer.w = normalizelogweights(self.SET_layer.logw)

		self.SNP_layer.pve=self.estimatePVE(self.SNP_layer,self.X)
		self.SET_layer.pve=self.estimatePVE(self.SET_layer,self.G)

		self.summarize_results(self.SNP_layer)
		self.summarize_results(self.SET_layer)

		if nonlinear_flag:
			self.residue = self.y - np.matmul(self.G, self.SET_layer.pip)
			self.W = []
			
			print("constructing kernelized features...")
			progbar = tf.keras.utils.Progbar(self.mask.shape[1])
			if self.epi_p:
				for i in np.arange(self.mask.shape[1]):
					indices = np.where(self.mask[:, i] == 1)[0]

					selected_X = self.X[:, indices]
					if (len(indices) <= 1):
						self.W.append(selected_X)
					else:
						all_combs = np.array(list(itertools.combinations(np.arange(selected_X.shape[1]), 2)))
						selected_comb_num = np.max([np.min([int(self.epi_p*len(all_combs)), self.max_epi]), 1])
						selected_combs = all_combs[np.random.choice(len(all_combs), selected_comb_num, replace=False)]

						self.W.append(np.column_stack([np.multiply(selected_X[:, idx[0]], selected_X[:, idx[1]]) for idx in selected_combs]))
					# self.W.append(RFF_gpytorch(gamma=0.1, n_components=self.n_components, seed=i).fit_transform(self.X[:, indices]))
					progbar.update(i+1)
			else:
				for i in np.arange(self.mask.shape[1]):
					indices = np.where(self.mask[:, i] == 1)[0]

					selected_X = self.X[:, indices]
					if (len(indices) <= 1):
						self.W.append(selected_X)
					else:
						self.W.append(np.column_stack([np.multiply(selected_X[:, j], selected_X[:, (j+1)]) for j in np.arange(len(indices)-1)]))
					# self.W.append(RFF_gpytorch(gamma=0.1, n_components=self.n_components, seed=i).fit_transform(self.X[:, indices]))
					progbar.update(i+1)


			self.W = np.column_stack(self.W)

			print("start optional layer: SET_nonlinear_layer...")

			self.SET_nonlinear_layer = HiddenLayer(self.W, self.residue, self.nModelsSET_nonlinear)
			self.SET_nonlinear_layer.train(self.W, self.residue, 10000)
			self.SET_nonlinear_layer.w = normalizelogweights(self.SET_nonlinear_layer.logw)

			self.SET_nonlinear_layer.pve=self.estimatePVE(self.SET_nonlinear_layer,self.W)
			self.summarize_results(self.SET_nonlinear_layer)
			return [self.SNP_layer, self.SET_layer, self.SET_nonlinear_layer]

		return [self.SNP_layer, self.SET_layer]

	def summarize_results(self,layer):
		layer.pip=np.sum(layer.w * layer.pip, axis=1)
		layer.kernel=np.sum(layer.w * layer.kernel, axis=1)

# X = np.loadtxt("Xtest2.txt")
# y = np.loadtxt("ytest2.txt")
# mask = np.loadtxt("masktest2.txt")


# bann=BANNs(X,y, mask, nModelsSNP=20, nModelsSET=20)
# [SNP_layer, SET_layer]=bann.run()
# print("PVE")
# print(SNP_layer.pve)
# print(SET_layer.pve)


# pips=SNP_layer.pip
# pips2=SET_layer.pip
# plt.scatter(np.arange(len(pips)), pips)
# plt.show()

# plt.scatter(np.arange(len(pips2)), pips2)
# plt.show()





