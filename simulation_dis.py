import numpy as np, os, sys, re, glob, subprocess, math, unittest, shutil, time, string, logging, gc
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
np.set_printoptions(precision=4)
from time import gmtime, strftime
from random import shuffle, choice, sample, choices
from itertools import product
from functools import partial
import inspect

base_folder = './'
try:
	os.mkdir(base_folder+'log')
	os.mkdir(base_folder+'agent_pool')
	os.mkdir(base_folder+'job_pool')
	os.mkdir(base_folder+'result_pool')
except:
	pass


current_time = strftime("%Y%m%d_%H%M%S", gmtime())

log_name = 'sim_{}_{}_{}_a{}.log'.format('data', sys.argv[2], sys.argv[3], sys.argv[4])
logging.basicConfig(filename=log_name, filemode='a', level=logging.DEBUG,
										format='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:  %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

class DummyIndv(object): pass

data = np.load('data.npz')
logging.info(data['adj_matrix'])
np.save('data.npy', data['goal'])
evoluation_goal = 'data.npy'

class Individual(object):
	def __init__(self, adj_matrix=None, scope=None, **kwargs):
		super(Individual, self).__init__()
		if adj_matrix is None:
			self.adj_matrix = kwargs['adj_func'](**kwargs)
		else:
			self.adj_matrix = adj_matrix
		self.scope = scope
		self.sparsityB = np.sum(self.adj_matrix[np.triu_indices(self.adj_matrix.shape[0], 1)]>0)
		self.parents = kwargs['parents'] if 'parents' in kwargs.keys() else None
		self.repeat = kwargs['evaluate_repeat'] if 'evaluate_repeat' in kwargs.keys() else 1
		self.iters = kwargs['max_iterations'] if 'max_iterations' in kwargs.keys() else 10000
		self.dim = self.adj_matrix.shape[0]
		self.adj_matrix[np.tril_indices(self.dim, -1)] = self.adj_matrix.transpose()[np.tril_indices(self.dim, -1)]
		adj_matrix_k = np.copy(self.adj_matrix)
		adj_matrix_k[adj_matrix_k==0] = 1
		self.present_elements = np.prod(np.diag(adj_matrix_k))
		self.actual_elements = np.sum([ np.prod(adj_matrix_k[d]) for d in range(self.dim) ])
		self.sparsity = self.actual_elements/self.present_elements

	def deploy(self, sge_job_id):
		try:
			path = base_folder+'/job_pool/{}.npz'.format(sge_job_id)
			np.savez(path, adj_matrix=self.adj_matrix, scope=self.scope, repeat=self.repeat, iters=self.iters)
			self.sge_job_id = sge_job_id
			return True
		except Exception as e:
			raise e

	def collect(self, fake_loss=False):
		if not fake_loss:
			try:
				path = base_folder+'/result_pool/{}.npz'.format(self.scope.replace('/', '_'))
				result = np.load(path)
				self.repeat_loss = result['repeat_loss']
				os.remove(path)
				return True
			except Exception:
				return False
		else:
			self.repeat_loss = [9999]*self.repeat
			return True

class Generation(object):
	def __init__(self, pG=None, name=None, **kwargs):
		super(Generation, self).__init__()
		self.name = name
		self.N_islands = kwargs['N_islands'] if 'N_islands' in kwargs.keys() else 1
		self.kwargs = kwargs
		self.out = self.kwargs['out']
		self.rank = self.kwargs['rank']
		self.size = self.kwargs['size']
		self.init_sparsity = kwargs['init_sparsity'] if 'init_sparsity' in kwargs.keys() else 0.8
		self.indv_to_collect = []
		self.indv_to_distribute = []
		if pG is not None:
			self.societies = {}
			for k, v in pG.societies.items():
				self.societies[k] = {}
				self.societies[k]['indv'] = \
						[ Individual( adj_matrix=indv.adj_matrix, parents=indv.parents,
													scope='{}/{}/{:03d}'.format(self.name, k, idx), **self.kwargs) \
						for idx, indv in enumerate(v['indv']) ]
				self.indv_to_distribute += [indv for indv in self.societies[k]['indv']]

		elif 'random_init' in kwargs.keys():
			self.societies = {}
			for n in range(self.kwargs['N_islands']):
				society_name = ''.join(choice(string.ascii_uppercase + string.digits) for _ in range(6))
				self.societies[society_name] = {}
				self.societies[society_name]['indv'] = [ \
						Individual(scope='{}/{}/{:03d}'.format(self.name, society_name, i), 
						adj_func=self.__random_adj_matrix__, **self.kwargs) \
						for i in range(self.kwargs['population'][n]) ]
				self.indv_to_distribute += [indv for indv in self.societies[society_name]['indv']]

	def __call__(self, **kwargs):
		try:
			self.__evaluate__()
			if 'callbacks' in kwargs.keys():
				for c in kwargs['callbacks']:
					c(self)
			self.__evolve__()
			return True
		except Exception as e:
			raise e

	def __random_adj_matrix__(self, **kwargs):
		if isinstance(self.out, list):
			adj_matrix = np.diag(self.out)
		else:
			adj_matrix = np.diag([self.out]*self.size)

		if self.init_sparsity < 0:
			connection = []
			real_init_sparsity = np.random.uniform(low=-self.init_sparsity, high=1.0)
			for i in range(np.sum(np.arange(self.size))):
				connection.append(int(np.random.uniform()>real_init_sparsity)*self.rank)
		else:
			connection = [ int(np.random.uniform()>self.init_sparsity)*self.rank for i in range(np.sum(np.arange(self.size)))]
		adj_matrix[np.triu_indices(self.size, 1)] = connection
		return adj_matrix

	def __evolve__(self):
		def mutation(indv, prob):
			dim = indv.adj_matrix.shape[0]
			elements = np.stack(np.triu_indices(dim, 1)).transpose()
			mask = np.random.uniform(size=elements.shape[0])<prob
			mutated_elements = tuple(map(tuple, elements[mask].transpose()))
			if mutated_elements:
				indv.adj_matrix[mutated_elements] = self.rank - indv.adj_matrix[mutated_elements]
				indv.adj_matrix[np.tril_indices(dim, -1)] = indv.adj_matrix.transpose()[np.tril_indices(dim, -1)]

		def immigration(islands, number=5):
			island_A, island_B = islands
			logging.info('immigration happend!')
			for _ in range(number):
				island_B.append(island_A.pop(0))
				island_A.append(island_B.pop(0))

		def elimination(island, threshold=80):
			island['rank'] = island['rank'][:threshold]
			island['indv'] = [island['indv'][i] for i in island['rank']]
			island['total'] = [island['total'][i] for i in island['rank']]

		def crossover(island, population, alpha=5):
			__adj_matrix__, __parents__ = [], []
			def propagation(couple, percent=0.5):
				adj_matrix_male = np.copy(couple[0].adj_matrix)
				adj_matrix_female = np.copy(couple[1].adj_matrix)

				dim = adj_matrix_male.shape[0]
				exchange_core = choice(list(range(dim)))

				exchange = adj_matrix_male[exchange_core]
				adj_matrix_male[exchange_core] = adj_matrix_female[exchange_core] 
				adj_matrix_female[exchange_core] = exchange

				adj_matrix_male[np.tril_indices(dim, -1)] = adj_matrix_male.transpose()[np.tril_indices(dim, -1)]
				adj_matrix_female[np.tril_indices(dim, -1)] = adj_matrix_female.transpose()[np.tril_indices(dim, -1)]

				__adj_matrix__.append(adj_matrix_male)
				__adj_matrix__.append(adj_matrix_female)
				__parents__.append((couple[0].scope[-13:], couple[1].scope[-13:]))
				__parents__.append((couple[0].scope[-13:], couple[1].scope[-13:]))

			indv, fitness = island['indv'], island['total']
			rank = np.argsort(fitness)
			# prob = [ 1.0/(1e-5+f)*alpha for f in fitness]		
			# p = [ np.exp(3/(1+k)) for k in range(len(indv)) ]
			p = [ np.maximum(np.log(float(sys.argv[4])/(0.01+k*5)), 0.01) for k in range(population) ]
			prob = np.zeros(len(indv))
			for idx, i in enumerate(rank): prob[i] = p[idx]
			for i in range(population//2): propagation(choices(indv, weights=prob, k=2))
			for i in range(population-len(indv)): indv.append(DummyIndv())
			for v, m, p in zip(indv, __adj_matrix__, __parents__): v.adj_matrix, v.parents = m, p

		# ELIMINATION
		if 'elimiation_threshold' in self.kwargs:
			for idx, (k, v) in enumerate(self.societies.items()):
				elimination(v, self.kwargs['elimiation_threshold'][idx])

		# IMMIRATION
		if 'immigration_prob' in self.kwargs:
			if np.random.uniform()<self.kwargs['immigration_prob']:
				immigration(sample([v['indv'] for k, v in self.societies.items()], k=2), self.kwargs['immigration_number'])

		# CROSSOVER
		if 'crossover_alpha' in self.kwargs:
			for idx, (k, v) in enumerate(self.societies.items()):
				crossover(v, self.kwargs['population'][idx], self.kwargs['crossover_alpha'])

		# MUTATION
		if 'mutation_prob' in self.kwargs:
			for k, v in self.societies.items():
				for indv in v['indv']:
					mutation(indv, self.kwargs['mutation_prob'])

	def __evaluate__(self):

		def score2rank(island, idx):
			sigmoid = lambda x : 1.0 / (1.0 + np.exp(-x))
			score = island['score']
			sparsity_score = [ s for s, l in score ]
			loss_score = [ l for s, l in score ]

			if 'fitness_func' in self.kwargs.keys():
				if isinstance(self.kwargs['fitness_func'], list):
					fitness_func = self.kwargs['fitness_func'][idx]
				else:
					fitness_func = self.kwargs['fitness_func']
			else:		
				fitness_func = lambda s, l: s+100*l
			
			total_score = [ fitness_func(s, l) for s, l in zip(sparsity_score, loss_score) ]

			island['rank'] = np.argsort(total_score)
			island['total'] = total_score

		# RANKING
		for idx, (k, v) in enumerate(self.societies.items()):
			v['score'] = [ (indv.sparsity ,np.min(indv.repeat_loss)) for indv in v['indv'] ]
			score2rank(v, idx)

	def distribute_indv(self, agent):
		if self.indv_to_distribute:
			indv = self.indv_to_distribute.pop(0)
			if np.log10(indv.sparsity)<1.0:
				agent.receive(indv)
				self.indv_to_collect.append(indv)
				logging.info('Assigned individual {} to agent {}.'.format(indv.scope, agent.sge_job_id))
			else:
				indv.collect(fake_loss=True)
				logging.info('Individual {} is killed due to its sparsity = {} / {}.'.format(indv.scope, np.log10(indv.sparsity), indv.sparsityB))

	def collect_indv(self):
		for indv in self.indv_to_collect:
			if indv.collect():
				logging.info('Collected individual result {}.'.format(indv.scope))
				self.indv_to_collect.remove(indv)

	def is_finished(self):
		if len(self.indv_to_distribute) == 0 and len(self.indv_to_collect) == 0:
			return True
		else:
			return False

class Agent(object):
	def __init__(self, **kwargs):
		super(Agent, self).__init__()
		self.kwargs = kwargs
		self.sge_job_id = self.kwargs['sge_job_id']

	def receive(self, indv):
		indv.deploy(self.sge_job_id)
		with open(base_folder+'/agent_pool/{}.POOL'.format(self.sge_job_id), 'a') as f:
			f.write(evoluation_goal)

	def is_available(self):
		return True if os.stat(base_folder+'/agent_pool/{}.POOL'.format(self.kwargs['sge_job_id'])).st_size == 0 else False

class Overlord(object):
	def __init__(self, max_generation=100, **kwargs):
		super(Overlord, self).__init__()
		self.dummy_func = lambda *args, **kwargs: None
		self.max_generation = max_generation
		self.current_generation = None
		self.previous_generation = None
		self.N_generation = 0
		self.kwargs = kwargs
		self.generation = kwargs['generation']
		self.generation_list = []
		self.available_agents = []
		self.known_agents = {}
		self.time = 0

	def __call_with_interval__(self, func, interval):
		return func if self.time%interval == 0 else self.dummy_func

	def __tik__(self, sec):
		# logging.info(self.time)
		self.time += sec
		time.sleep(sec)

	def __check_available_agent__(self):
		self.available_agents.clear()
		agents = glob.glob(base_folder+'/agent_pool/*.POOL')
		agents_id = [ a.split('/')[-1][:-5] for a in agents ]

		for aid in list(self.known_agents.keys()):
			if aid not in agents_id:
				logging.info('Dead agent id = {} found!'.format(aid))
				self.known_agents.pop(aid, None)

		for aid in agents_id:
			if aid in self.known_agents.keys():
				if self.known_agents[aid].is_available():
					self.available_agents.append(self.known_agents[aid])
			else:
				self.known_agents[aid] = Agent(sge_job_id=aid)
				logging.info('New agent id = {} found!'.format(aid))

	def __assign_job__(self):
		self.__check_available_agent__()
		if len(self.available_agents)>0:
			for agent in self.available_agents:
				self.current_generation.distribute_indv(agent)

	def __collect_result__(self):
		self.current_generation.collect_indv()

	def __report_agents__(self):
		logging.info('Current number of known agents is {}.'.format(len(self.known_agents)))
		logging.info(list(self.known_agents.keys()))

	def __report_generation__(self):
		logging.info('Current length of indv_to_distribute is {}.'.format(len(self.current_generation.indv_to_distribute)))
		logging.info('Current length of indv_to_collect is {}.'.format(len(self.current_generation.indv_to_collect)))
		logging.info([(indv.scope, indv.sge_job_id) for indv in self.current_generation.indv_to_collect])

	def __generation__(self):
		if self.N_generation > self.max_generation:
			return False
		else:
			if self.current_generation is None:
				self.current_generation = self.generation(name='generation_init', **self.kwargs)
				self.current_generation.indv_to_distribute = []

			if self.current_generation.is_finished():
				if self.previous_generation is not None:
					self.current_generation(**self.kwargs)
				self.N_generation += 1
				self.previous_generation = self.current_generation
				self.current_generation = self.generation(self.previous_generation, 
														name='generation_{:03d}'.format(self.N_generation), **self.kwargs)

			return True

	def __call__(self):
		while self.__generation__():
			self.__call_with_interval__(self.__check_available_agent__, 4)()
			self.__call_with_interval__(self.__assign_job__, 4)()
			self.__call_with_interval__(self.__collect_result__, 4)()
			self.__call_with_interval__(self.__report_agents__, 180)()
			self.__call_with_interval__(self.__report_generation__, 160)()
			self.__tik__(2)

def score_summary(obj):
	logging.info('===== {} ====='.format(obj.name))

	for k, v in obj.societies.items():
		logging.info('===== ISLAND {} ====='.format(k))

		for idx, indv in enumerate(v['indv']):
			if idx == v['rank'][0]:
				logging.info('\033[31m{} | {:.3f} | {} | {:.5f} | {}\033[0m'.format(indv.scope, np.log10(indv.sparsity), [ float('{:0.4f}'.format(l)) for l in indv.repeat_loss ], v['total'][idx], indv.parents))
				logging.info(indv.adj_matrix)
			else:
				logging.info('{} | {:.3f} | {} | {:.5f} | {}'.format(indv.scope, np.log10(indv.sparsity), [ float('{:0.4f}'.format(l)) for l in indv.repeat_loss ], v['total'][idx], indv.parents))
				logging.info(indv.adj_matrix)

if __name__ == '__main__':
	pipeline = Overlord(		# GENERATION PROPERTIES
													max_generation=30, generation=Generation, random_init=True,
													# ISLAND PROPERTIES
													N_islands=1, population=[int(sys.argv[2])], 
													# INVIDUAL PROPERTIES
													size=4, rank=2, out=2, init_sparsity=-0.00001,
													# EVALUATION PROPERTIES
													evaluate_repeat=2, max_iterations=10000,
													fitness_func=[ lambda s,l: s+l*50],
													#
													# EVOLUTION PROPERTIES
													elimiation_threshold=[int(sys.argv[3])], immigration_prob=0, immigration_number=5,
													crossover_alpha=1, mutation_prob=0.05,
													# FOR COMPUTATION
													callbacks=[score_summary])
	pipeline()