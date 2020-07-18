import re, os, sys, math, unittest
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import random
from random import shuffle, choice
from itertools import product
# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import connected_components
import operator 
import numpy as np
np.set_printoptions(precision=4)
# from decorator import decorator
import operator
from functools import reduce
# from scipy.io import loadmat
from tensorflow.python.framework.ops import Tensor

def prod(iterable):
	return reduce(operator.mul, iterable, 1)

def letter_range(n):
	for c in range(97, 97+n):
		yield chr(c)


class RealTensor(object):
	def __init__(self, name='no_name', from_variable=None, shape=[1, 2, 3, 4], trainable=True,
										initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0), identity=None):

		if from_variable is not None:
			self.name = name
			self.identity = identity
			self.shape = from_variable.get_shape().as_list()
			self.tensor = tf.identity(from_variable, name=self.name)
		else:
			self.name = name
			self.identity = identity
			self.shape = shape
			self.initializer = initializer
			if isinstance(self.initializer, Tensor):
				self.tensor = tf.get_variable(name = self.name,	initializer = self.initializer, trainable=trainable)
			else:
				self.tensor = tf.get_variable(name = self.name, shape = self.shape,
																			initializer = self.initializer, trainable=trainable)

	def __call__(self):
		return self.tensor

class TensorNetwork(object):
	def __init__(self, adj_matrix, name_list=None, initializer_list=None, trainable_list=None, scope='TensorNetwork'):
		self.shape = adj_matrix.shape
		assert self.shape[0] == self.shape[1], 'adj_matrix must be a square matrix.'
		self.dim = self.shape[0]
		self.adj_matrix = np.empty(self.shape, dtype=object)
		self.scope = scope
		self.output_count = 0
		self.output_order = []
		for i in np.diag(adj_matrix):
			if i == 0:
				self.output_order.append([])
			else:
				self.output_order.append([self.output_count])
				self.output_count += 1

		self.id_matrix = np.empty(self.shape, dtype=object)

		tril_idx = np.tril_indices(self.dim, -1)
		if np.sum(adj_matrix[tril_idx]) == 0:
			adj_matrix[tril_idx] += adj_matrix[(tril_idx[1], tril_idx[0])]

		# graph = np.copy(adj_matrix)
		# graph[np.diag_indices(self.dim)] = 0
		# graph[graph>0] = 1
		# graph = csr_matrix(graph)
		# n_components = connected_components(csgraph=graph, directed=False, return_labels=False)
		# if not n_components == 1:
		# 	print ('The network is seperated by {} parts.'.format(n_components))


		for idx in np.ndindex(self.shape):
			self.adj_matrix[idx] = [ adj_matrix[idx] ]

			if self.adj_matrix[idx][0] == 0:
				self.adj_matrix[idx].clear()
				# if idx[0] == idx[1]:
				# 	self.adj_matrix[idx].append(1)

		if name_list is not None:
			assert self.dim == len(name_list), 'Length of name_list does not match number of cores.'
			self.name_list = name_list
		else:
			self.name_list = list(letter_range(self.dim))

		if trainable_list is None:
			trainable_list = [True] * self.dim

		if initializer_list is None:
			initializer_list = [tf.random_normal_initializer(mean=0.0, stddev=1.0)] * self.dim

		with tf.variable_scope(self.scope):
			self.cores = [ RealTensor(name=self.name_list[t], shape=list(filter((0).__ne__, adj_matrix[t].tolist())),
										trainable=trainable_list[t], initializer=initializer_list[t]) for t in range(self.dim) ]

	def __repr__(self):
		return self.reduction()

	def __add__(self, TN_b):
		return self.reduction() + N_b.reduction()

	def __sub_(self, TN_b):
		return self.reduction() - N_b.reduction()

	def __mul_(self, TN_b):
		return __tf_matmul__(self.reduction(), N_b.reduction())

	def giff_cores(self):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

	def __outter_product__(self):
		tar = tf.reshape(self.cores[1](), [1, -1])
		des = tf.reshape(self.cores[0](), [-1, 1])
		reduced_core = tf.reshape(tf.matmul(des, tar), [-1])

		self.adj_matrix[0][0] = self.adj_matrix[0][0] + self.adj_matrix[1][1]
		self.output_order[1] = self.output_order[0] + self.output_order[1]
		self.output_order.pop(0)

		self.adj_matrix = np.delete(self.adj_matrix, 1, 0)
		self.adj_matrix = np.delete(self.adj_matrix, 1, 1)

		self.dim -= 1

		reduced_core_name = self.name_list[1]+self.name_list[0]
		self.cores.pop(1)
		self.cores.pop(0)
		self.cores.insert(0, RealTensor(name=reduced_core_name, from_variable=reduced_core))
		self.name_list.pop(1)
		self.name_list.pop(0)
		self.name_list.insert(0, reduced_core_name)

	def __reduce_cores__(self, target_destination):
		target, destination = target_destination
		for idx in np.ndindex((self.dim, self.dim)):
			if len(self.adj_matrix[idx]) > 0:
				self.id_matrix[idx] = self.cores[idx[1]].name[0]
			else:
				self.id_matrix[idx] = ''
		for idx, c in enumerate(self.cores):
			c.identity = reduce(operator.add, self.id_matrix[idx])

		# print (self.id_matrix)

		# print (self.adj_matrix)
		target_shape = [ int(np.prod(i)) for i in self.adj_matrix[target].tolist() ]
		destination_shape = [ int(np.prod(i)) for i in self.adj_matrix[destination].tolist() ]
		# print (self.cores[target](), target_shape)
		# print (self.cores[destination](), destination_shape)
		tar = tf.reshape(self.cores[target](), target_shape)
		des = tf.reshape(self.cores[destination](), destination_shape)

		tar_trans_list, des_trans_list = list(range(self.dim)), list(range(self.dim))
		tar_trans_list = tar_trans_list + [tar_trans_list.pop(destination)]
		des_trans_list = [des_trans_list.pop(target)] + des_trans_list 
		tar, des = tf.transpose(tar, tar_trans_list), tf.transpose(des, des_trans_list)
		reduced_core = self.__tf_matmul__(tar, des)
		# print (reduced_core)

		reduced_trans_list = list(range(self.dim*2-2))
		reduce_trans_list_des = reduced_trans_list.pop(destination+self.dim-2)
		reduce_trans_list_tar = reduced_trans_list.pop(target)
		reduced_trans_list_len = len(reduced_trans_list)//2
		reduced_trans_list = [ [ reduced_trans_list[i], reduced_trans_list[i+reduced_trans_list_len] ] for i in range(reduced_trans_list_len)]
		reduced_trans_list.insert(destination-1, [reduce_trans_list_tar, reduce_trans_list_des] )
		reduced_trans_list = [ k for j in reduced_trans_list for k in j]
		# for i in range(self.dim-1):
		# 	reduced_trans_list += [i, i+self.dim-1]
		# print (reduced_trans_list)
		reduced_core = tf.transpose(reduced_core, reduced_trans_list)
		reduced_core = tf.squeeze(reduced_core)

		self.adj_matrix[destination, destination] = self.adj_matrix[target, target] + self.adj_matrix[destination, destination]
		self.output_order[destination] = self.output_order[target] + self.output_order[destination]
		self.output_order.pop(target)

		inherit = list(range(self.dim))
		inherit.remove(target)
		inherit.remove(destination)

		# print (self.adj_matrix)
		for i in inherit:
			self.cores[i].tensor = tf.reshape(self.cores[i](), [ int(np.prod(z)) for z in self.adj_matrix[i].tolist() if int(np.prod(z))>1 ])
			self.adj_matrix[destination][i] = self.adj_matrix[target][i] + self.adj_matrix[destination][i]
			self.adj_matrix[i][destination] = self.adj_matrix[i][target] + self.adj_matrix[i][destination]
			self.id_matrix[i][destination] = self.id_matrix[i][target] + self.id_matrix[i][destination]
			self.id_matrix[i][target] = ''

		self.adj_matrix = np.delete(self.adj_matrix, target, 1)

		# print (self.adj_matrix)
		for i in inherit:
			if self.cores[i].identity != reduce(operator.add, self.id_matrix[i]):
				# print (i, self.cores[i](), self.cores[i].identity, reduce(operator.add, self.id_matrix[i]))
				id_tran_tar = list(reduce(operator.add, self.id_matrix[i]))
				id_tran_des = list(self.cores[i].identity)
				id_tran = [ id_tran_des.index(i) for i in id_tran_tar]
				# print (id_tran)
				self.cores[i].tensor = tf.transpose(self.cores[i].tensor, id_tran)
				# print (i, self.cores[i](), self.cores[i].identity, reduce(operator.add, self.id_matrix[i]))

		self.adj_matrix = np.delete(self.adj_matrix, target, 0)
		self.id_matrix = np.delete(self.id_matrix, target, 0)
		self.id_matrix = np.delete(self.id_matrix, target, 1)

		self.dim -= 1

		reduced_core_name = self.name_list[target]+self.name_list[destination]
		self.cores.pop(destination)
		self.cores.pop(target)
		self.cores.insert(destination-1, RealTensor(name=reduced_core_name, from_variable=reduced_core))
		self.name_list.pop(destination)
		self.name_list.pop(target)
		self.name_list.insert(destination-1, reduced_core_name)

		# print ([ c() for c in self.cores])
		# print ('=============')

	def __tf_matmul__(self, A, B):
		A_shape = A.get_shape().as_list()
		B_shape = B.get_shape().as_list()
		A = tf.reshape(A, [-1, A_shape[-1]])
		B = tf.reshape(B, [B_shape[0], -1])

		O_shape = A_shape[:-1] + B_shape[1:]
		O = tf.matmul(A, B)
		O = tf.reshape(O, O_shape)

		return O

	def __pre_calculation__(self, target_destination):
		target, destination = target_destination
		# print (self.adj_matrix)

		adj_matrix_k = np.copy(self.adj_matrix)
		N_elements = []
		for d in range(self.dim):
			N_elements.append(np.prod([ np.prod(adj_matrix_k[d][b]) for b in range(self.dim) ]))

		N_destination = N_elements.pop(destination)
		N_target = N_elements.pop(target)
		N_target_destination = N_destination*N_target/np.square(np.prod(adj_matrix_k[target][destination]))

		return np.sum(N_elements)+N_target_destination


	def reduction(self, random=True):
		while len(self.cores) > 1:
			triu_indices = np.triu_indices(self.dim, 1)

			if len(np.sum(self.adj_matrix[triu_indices])) == 0:
				self.__outter_product__()

			else:
				if random:
					connctions = []
					for i in range(triu_indices[0].shape[0]):
						if len(self.adj_matrix[(triu_indices[0][i], triu_indices[1][i])]) > 0:
							connctions.append((triu_indices[0][i], triu_indices[1][i]))

					c = choice(connctions)
					self.__reduce_cores__(c)

				else:
					c, cc = None, None
					for i in range(triu_indices[0].shape[0]):
						if len(self.adj_matrix[(triu_indices[0][i], triu_indices[1][i])]) > 0:
							N = self.__pre_calculation__((triu_indices[0][i], triu_indices[1][i]))
							if c is None: 
								c = (triu_indices[0][i], triu_indices[1][i])
							if cc is None: 
								cc = N
							if N < cc:
								c = (triu_indices[0][i], triu_indices[1][i])
								cc = N

					self.__reduce_cores__(c)

		output = tf.reshape(self.cores[0](), self.adj_matrix[0][0])

		# print (self.output_order)
		self.output_order = self.output_order[0]
		output_trans = np.zeros((self.output_count,), dtype=int)
		for i in range(self.output_count):
			output_trans[self.output_order[i]] = i
		# print(output_trans)
		output = tf.transpose(output, output_trans)

		output = tf.squeeze(output)
		return tf.identity(output, name='output')

	def opt_opeartions(self, opt, loss):
		return opt.minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

if __name__ == '__main__':

	# An example of TR
	# TR_A = np.array([ [ 2,  6,  0,  0,  0,  0,  0],
	# 									[ 0,  3,  0,  0,  0,  0,  0],
	# 									[ 0,  0,  5,  6,  0,  0,  0],
	# 									[ 0,  0,  0,  7,  0,  0,  0],
	# 									[ 0,  0,  0,  0, 11,  6,  0],
	# 									[ 0,  0,  0,  0,  0, 13,  6],
	# 									[ 0,  0,  0,  0,  0,  0, 17] ])

	# tr_a = TensorNetwork(TR_A)
	# output_a = tr_a.reduction(False)
	# goal_a = tf.random_normal(shape=output_a.get_shape().as_list(), seed=100)
	# mse_loss = tf.reduce_mean(tf.square(output_a - goal_a))

	# step = tr_a.opt_opeartions(tf.train.AdamOptimizer(0.0001), mse_loss)

	# sess = tf.Session()
	# sess.run(tf.global_variables_initializer())
	# for i in range(10000):
	# 	_, loss = sess.run([step, mse_loss])
	# 	if (i+1)%100 == 0:
	# 		print(loss)

	# An exmple of random tensor network
	# def generate_random_TN(output_shape, num_cores, max_connection, max_dim):
	# 	matrix = np.diag(output_shape + [0]*(num_cores - len(output_shape)))

	# 	idx = random.choices(np.array(np.triu_indices(num_cores, 1)).transpose(), k=random.randint(2, max_connection))
	# 	for i in idx:
	# 		i = tuple(i)
	# 		print(i)
	# 		matrix[i] = random.randint(2, max_dim)

	# 	return matrix

	# a = generate_random_TN([2,3,5,7], 7, 15, 10)

	# print (a)

	# CP CP CP
	def generate_CPTucker(output_shape, num_cores, dim):
		if isinstance(output_shape, list):
			output_shape = [0] + output_shape
		else:
			output_shape = [0] + [output_shape] * num_cores

		if isinstance(dim, list):
			dim = np.array([0] + dim)
		else:
			dim = np.array([0] + [dim] * num_cores)
				
		matrix = np.diag(output_shape)
		matrix[0] = dim

		return matrix

	# cp = loadmat('data_CP.mat')
	# U, X = cp['U'][0], cp['X'][0]

	# sess = tf.Session()
	# for i in range(48):
	# 	ground_turth = X[i]

	# 	num_cores = U[i].shape[1]
	# 	output_shape = U[i][0][0].shape[0]
	# 	dim = U[i][0][0].shape[1]

	# 	adj_matrix = generate_CPTucker(output_shape, num_cores, dim)

	# 	cp_diag = np.zeros([dim]*num_cores, dtype=np.float32)
	# 	np.fill_diagonal(cp_diag, 1)

	# 	# init_np = [cp_diag] + [ np.array(U[i][0][n]).transpose() for n in range(num_cores)]
	# 	# init_tf = [ tf.convert_to_tensor(i) for i in init_np ]

	# 	init_tf = [ tf.convert_to_tensor(cp_diag) ] + [tf.random_normal_initializer(mean=0.0, stddev=0.1)]*num_cores

	# 	tn = TensorNetwork(adj_matrix, initializer_list=init_tf, scope='TensorNetwork_{}'.format(i),
	# 																	trainable_list = [False] + [True] * num_cores)

	# 	output = tn.reduction(False)		

	# 	gt = tf.convert_to_tensor(ground_turth)
	# 	mse_loss = tf.losses.mean_squared_error(gt, output)
	# 	step = tn.opt_opeartions(tf.train.AdamOptimizer(0.001), mse_loss)

	# 	sess.run(tf.global_variables_initializer())
	# 	for i in range(100000):
	# 		_, loss = sess.run([step, mse_loss])
	# 		if loss < 1e-7:
	# 			break

	# 	print (i, loss)

		# o = sess.run(output)

		# diff = np.mean(np.square(ground_turth - o))

		# print (diff)

		# print (o)
		# print (ground_turth)

		# print (adj_matrix)

	# Tucker Tucker
	# tucker = loadmat('data_Tucker.mat')
	# G, U, X = tucker['G'][0], tucker['U'][0], tucker['X'][0]

	# sess = tf.Session()
	# for i in range(22,23):
	# 	ground_turth = X[i]

	# 	num_cores = U[i].shape[1]
	# 	output_shape = U[i][0][0].shape[0]
	# 	dim = U[i][0][0].shape[1]

	# 	adj_matrix = generate_CPTucker(output_shape, num_cores, dim)
	# 	# print (adj_matrix)

	# 	init_np = [G[i]] + [ np.array(U[i][0][n]).transpose() for n in range(num_cores)]
	# 	init_tf = [ tf.convert_to_tensor(i) for i in init_np ]


	# 	# init_tf = [tf.random_normal_initializer(mean=0.0, stddev=0.1)]*(num_cores+1)

	# 	# tn = TensorNetwork(adj_matrix, initializer_list=init_tf, scope='TensorNetwork_{}'.format(i),
	# 	# 																trainable_list = [True]*(num_cores+1))


	# 	tn = TensorNetwork(adj_matrix, initializer_list=init_tf, scope='TensorNetwork_{}'.format(i))

	# 	output = tn.reduction(False)		

	# 	gt = tf.convert_to_tensor(ground_turth)
	# 	mse_loss = tf.losses.mean_squared_error(gt, output)
	# 	step = tn.opt_opeartions(tf.train.AdamOptimizer(0.001), mse_loss)

	# 	sess.run(tf.global_variables_initializer())
	# 	# for i in range(10000):
	# 	# 	_, loss = sess.run([step, mse_loss])
	# 	# 	if loss < 1e-7:
	# 	# 		break

	# 	# print (i, loss)

	# 	o = sess.run(output)
	# 	diff = np.mean(np.square(ground_turth - o))

	# 	print (diff)

		# from itertools import permutations

		# for idx in permutations(range(3)):
		# 	o = np.transpose(o, idx)

		# 	diff = np.mean(np.square(ground_turth - o))

		# 	print (diff)


	# OUTTER PRODUCT

	# outter = loadmat('rank1.mat')
	# U, X = outter['U'][0], outter['X'][0]

	# sess = tf.Session()
	# for i in range(9):
	# 	ground_turth = X[i]
	# 	num_cores = len(X[i].shape)

	# 	adj_matrix = np.diag(X[i].shape)

	# 	init_np = [ np.squeeze(np.array(U[i][0][n])) for n in range(num_cores)]
	# 	init_tf = [ tf.convert_to_tensor(i) for i in init_np ]

	# 	tn = TensorNetwork(adj_matrix, initializer_list=init_tf, scope='TensorNetwork_{}'.format(i))
	# 	output = tn.reduction(True)
	# 	sess.run(tf.global_variables_initializer())
	# 	o = sess.run(output)

	# 	diff = np.mean(np.square(ground_turth - o))

	# 	print (diff)

	sess = tf.Session()
	adjm = np.array([[0,2,2,2,2,0,0,0,0],
									 [0,11,2,2,2,2,2,2,0],
									 [0,0,12,2,2,0,0,2,0],
									 [0,0,0,0,2,2,0,0,2],
									 [0,0,0,0,0,0,2,2,2],
									 [0,0,0,0,0,0,0,2,2],
									 [0,0,0,0,0,0,13,2,2],
									 [0,0,0,0,0,0,0,14,2],
									 [0,0,0,0,0,0,0,0,15]] , dtype=int)


	TN = TensorNetwork(adjm)
	ot = TN.reduction(False)
	sess.run(tf.global_variables_initializer())
	sess.run(ot)
	# opt = tf.train.AdamOptimizer(0.001)
	# # opt = tf.train.GradientDescentOptimizer(0.5)

	# goal = tf.convert_to_tensor(evoluation_goal)
	# goal_square_norm = tf.convert_to_tensor(evoluation_goal_square_norm)

	# indv_mse_loss = tf.reduce_mean(tf.square(ot - goal)) / goal_square_norm
	# step = opt.minimize(indv_mse_loss)