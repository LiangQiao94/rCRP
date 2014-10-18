'''
rCRP implementation by Joon Hee Kim (joonheekim@gmail.com)
'''

import re, gensim, pickle, logging, random, time, math, os
import numpy as np
from copy import copy
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
data_dir = '/Users/joonheekim/Projects/rCRP/data/'
corpus_dir = '/Users/joonheekim/Projects/rCRP/data/corpus/'
utils_dir = '/Users/joonheekim/Projects/rCRP/data/utils/'
exp_dir = '/Users/joonheekim/Projects/rCRP/experiment/'

class Table:
	def __init__(self):
		self.words = []
		self.topic = None

class Topic:
	def __init__(self, num_word, beta, level):
		# self.index = index
		self.table_count = 0
		self.children = []
		self.parent = None
		self.phi = np.ones((num_word)) * beta
		self.phi_sum = beta * num_word
		self.level = level

class Document:
	def __init__(self, docID, time, words):
		self.docID = docID
		self.time = time
		self.words = words
		self.word2table = [None] * len(words)
		self.tables = []

class Model:
	def __init__(self, alpha, beta, gamma, data_type, iteration, no_below, no_above, delta, depth_limit):
		self.topics = {}
		self.alpha = alpha 		# prob to create new table
		self.gamma = gamma		# prob to create new topic
		self.beta = beta # topic smoothing factor
		self.delta = delta
		self.data_type = data_type
		self.iteration = iteration
		self.no_below = no_below
		self.no_above = no_above
		self.depth_limit = depth_limit
		self.param = data_type + '_' + str(alpha) + '_' + str(beta) + '_' + str(gamma) + '_'  + str(delta) + '_' + str(iteration) + str(depth_limit)
		# load non-topical base topic

	def load_data(self):
		self.dictionary = gensim.corpora.dictionary.Dictionary.load(utils_dir + self.data_type + '_' + str(self.no_below) + '_' + str(self.no_above) + '.dic')
		self.num_word = len(self.dictionary)
		self.corpus = []
		length = 0
		num_doc = 0

		logging.info('reading data')	
		f = open(corpus_dir + self.data_type + '_' + str(self.no_below) + '_' + str(self.no_above) + '/' + self.data_type + '.txt', 'r')
		for line in f:
			words = [int(word) for word in line.strip().split()]
			doc = Document(None, None, words)
			self.corpus.append(doc)
			num_doc += 1
			length += len(words)
		f.close()
		logging.info('average document length:' + str(length / float(num_doc)))

	def run(self):
		for i in range(self.iteration):
			for corpus in self.corpus:
		 		print 'running data for',year_count
				for document in corpus:
					self.process_document(document)
			self.print_topic_count(i)
			self.print_table_count(i)
			self.print_state(i)
		

	def top_words(self, vector, n):
		vector = copy(vector)
		result = ''
		for i in range(n):
			argmax = np.argmax(vector)
			value = vector[argmax]
			vector[argmax] = -1
			result += self.dictionary[argmax]
			result += '\t'
			result += ("%.3f"%value)
			result += '\t'
		return result

	def choose(self, roulette):
		total = sum(roulette)
		arrow = total * random.random()
		for i in range(len(roulette)):
			if arrow < roulette[i]:
				return i
			arrow -= roulette[i]
		print 'error in choose'
		print roulette
		return -1
			
	# def print_state(self, i):
	# 	if not os.path.isdir(exp_dir + self.param):
	# 		os.mkdir(exp_dir + self.param)
	# 	write_file = open(exp_dir + self.param + '/' + str(i) + '.txt', 'w')
	# 	for topic in self.topics.keys():
	# 		write_file.write(str(topic.table_count) + '\t')
	# 		write_file.write(str(int(topic.phi_sum - self.num_word * self.beta)) + '\t')
	# 		write_file.write(self.top_words(topic.phi, 10) + '\n')
	# 		write_file.write('\n')
	# 		for child_topic in self.topics[topic]:
	# 			write_file.write(str(child_topic.table_count) + '\t')
	# 			write_file.write(str(int(child_topic.phi_sum - self.num_word * self.beta)) + '\t')
	# 			write_file.write(self.top_words(child_topic.phi, 10) + '\n')
	# 		write_file.write('\n\n\n')
	# 	write_file.close()
	# 	print self.param

	# def print_topic_count(self, i):
	# 	# print '\n'*2
	# 	if random.random() < 0:
	# 		for topic, children in self.topics.items():
	# 			for child in children:
	# 				print child.table_count
	# 			print '\n'

	# 	num = 0
	# 	for topic, children in self.topics.items():
	# 		num += len(children)

	def print_table_count(self, i):
		num = 0
		num_word = 0
		num_doc = 0
		for document in self.corpus:
			num += len(document.tables)
			num_doc += 1
		logging.info('average table count' + str(float(num) / num_doc))

	def process_document(self, document):
		# self.print_topic_count()
		if len(document.tables) == 0:
			random.shuffle(document.words)

		# table assignment
		for i in range(len(document.words)):
			word = document.words[i]
			# de-assignment
			old_table = document.word2table[i]
			if old_table == None:
				pass
			else:
				# print 'de-assignment'
				old_table.words.remove(word)
				old_topic = old_table.topic
				old_topic.phi[word] -= 1
				old_topic.phi_sum -= 1

				parent_topic = old_topic.parent
				while(parent_topic != None):
					parent_topic.phi[word] -= 1
					parent_topic.phi_sum -= 1
					parent_topic = parent_topic.parent
				
				if len(old_table.words) == 0:
					document.tables.remove(old_table)
					old_topic.table_count -= 1
					
					parent_topic = old_topic.parent
					while(parent_topic != None):
						parent_topic.table_count -= 1
						parent_topic = parent_topic.parent
					
					
					parent_topic = old_topic.parent
					curr_topic = old_topic
					while(parent_topic != None):
						if curr_topic.table_count == 0:
							parent_topic.children.remove(curr_topic)
							curr_topic = parent_topic
							parent_topic = parent_topic.parent
						else:
							break

			# assignment
			roulette = np.zeros((len(document.tables) + 1))
			for j in range(len(document.tables)):
				table = document.tables[j]
				roulette[j] = (table.topic.phi[word] / table.topic.phi_sum) * len(table.words)
				# if roulette[j] < 0:
				# print j, table.topic.phi[word], table.topic.phi_sum, len(table.words)
			roulette[-1] = self.alpha / self.num_word
			new_table_index = self.choose(roulette)
			# print roulette, new_table_index
			if new_table_index == -1:
				print 'error 1'
				exit(-1)
			if new_table_index == len(document.tables):
				new_table = Table()
				document.tables.append(new_table)
				new_topic = self.get_topic_for_table(new_table)
				new_table.topic = new_topic
				new_topic.table_count += 1
				new_topic.parent.table_count += 1
			else:
				new_table = document.tables[new_table_index]
				new_topic = new_table.topic
			new_table.words.append(word)
			new_table.topic.phi[word] += 1
			new_table.topic.phi_sum += 1
			new_table.topic.parent.phi[word] += 1
			new_table.topic.parent.phi_sum += 1
			document.word2table[i] = new_table

		# topic assignment
		for i in range(len(document.tables)):
			# de-assignment
			table = document.tables[i]
			old_topic = table.topic
			parent_topic = old_topic.parent

			for word in table.words:
				old_topic.phi[word] -= 1					
			old_topic.phi_sum -= len(table.words)
			old_topic.table_count -= 1									
			if old_topic.table_count == 0:
				parent_topic.children.remove(old_topic)

			while(parent_topic != None):
				for word in table.words:
					parent_topic.phi[word] -= 1					
				parent_topic.phi_sum -= len(table.words)
				parent_topic.table_count -= 1									
				if parent_topic.table_count == 0:
					parent_topic.parent.children.remove(parent_topic)
				parent_topic = parent_topic.parent

			new_topic = self.get_topic_for_table(table)
			table.topic = new_topic
			
			for word in table.words:
				new_topic.phi[word] += 1
			new_topic.phi_sum += len(table.words)
			new_topic.table_count += 1

			parent_topic = new_topic.parent
			while(parent_topic != None):
				for word in table.words:
					parent_topic.phi[word] += 1
				parent_topic.phi_sum += len(table.words)
				parent_topic.table_count += 1
				parent_topic = parent_topic.parent
		
	def get_topic_for_table(self, table):	
		word_limit = 100


		
		# choose base topic
		roulette = np.zeros((len(self.topics) + 1))
		for i in range(len(self.topics)):
			topic = self.topics.keys()[i]
			roulette[i] =  topic.table_count
			for word in table.words[:word_limit]:
				roulette[i] *= (topic.phi[word] / topic.phi_sum * self.num_word)
		roulette[-1] = self.gamma1 # / math.pow(self.num_word, len(table.words))
		# print roulette
		base_topic_index = self.choose(roulette)
		if base_topic_index == -1:
			print 'error 2'
			print 'len(table.words):', len(table.words)
			exit(-1)
		if base_topic_index == len(self.topics):
			# create new base topic
			base_topic = Topic(self.num_word, self.beta)
			self.topics[base_topic] = []
		else:
			base_topic = self.topics.keys()[base_topic_index]

		# choose sub topic
		roulette = np.zeros((len(self.topics[base_topic]) + 1))
		# print table.words
		for i in range(len(self.topics[base_topic])):
			topic = self.topics[base_topic][i]
			roulette[i] =  topic.table_count
			for word in table.words[:word_limit]:
				roulette[i] *= (topic.phi[word] / topic.phi_sum * self.num_word)
		roulette[-1] = self.gamma2 # / math.pow(self.num_word, len(table.words))
		sub_topic_index = self.choose(roulette)
		if sub_topic_index == -1:
			print 'error 3'
			print 'len(table.words):', len(table.words)			
			exit(-1)
		if sub_topic_index == len(self.topics[base_topic]):
			# create new base topic
			sub_topic = Topic(self.num_word, self.beta)
			sub_topic.parent = base_topic
			sub_topic.phi = copy(sub_topic.parent.phi) * self.delta
			sub_topic.phi_sum = copy(sub_topic.parent.phi_sum) * self.delta
			self.topics[base_topic].append(sub_topic)
		else:
			sub_topic = self.topics[base_topic][sub_topic_index]
		# sub_topic.table_count += 1
		return sub_topic

def test_nips():
	alpha = 1.0
	beta = 0.1
	gamma = 0.2
	data_type = 'nips'
	iteration = 1000
	no_below = 5
	no_above = 0.2
	delta = 0.01
	depth_limit = 4
	model = Model(alpha, beta, gamma, data_type, iteration, no_below, no_above, delta, depth_limit)

	model.load_data()
	# model.run()

test_nips()