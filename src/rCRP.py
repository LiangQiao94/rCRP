'''
rCRP implementation by Joon Hee Kim (joonheekim@gmail.com)

phi = topics
phi_sum = sum of phi
alpha = parameter for creating new table
beta = parameter for topic smoothing
gamma1 = parameter for creating new topic
gamma2 = parameter for creating new topic
where gamma = gamma1 * gamma2^topic_level (root = level 0)
delta = parameter for copying phi from parent topics
no_below = same as in gensim library
no_above = same as in gensim library

data format for corpus should consists of a file where each line is a document, and each token is separated by a tab
an example is in /corpus/nips_5_0.2/nips.txt
where 5 = no_below and 0.2 = no_above
'''

import re, gensim, pickle, logging, random, time, math, os, bisect
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
		self.big_table_count = 0
		self.small_table_count = 0
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
	def __init__(self, alpha, beta, gamma1, gamma2, data_type, iteration, no_below, no_above, delta, depth_limit):
		self.alpha = alpha 	
		self.gamma1 = gamma1
		self.gamma2 = gamma2
		self.beta = beta # topic smoothing factor
		self.delta = delta
		self.data_type = data_type
		self.iteration = iteration
		self.no_below = no_below
		self.no_above = no_above
		self.depth_limit = depth_limit
		self.param = data_type + '_' + str(alpha) + '_' + str(beta) + '_' + str(gamma1) + '_' + str(gamma2) + '_' + str(delta) + '_' + str(depth_limit)

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

		self.topics = {}		# {level: topics}
		root_topic = Topic(self.num_word, self.beta, 0)
		root_topic.big_table_count += 1
		root_topic.small_table_count += 1
		for i in range(self.depth_limit):
			self.topics[i] = []
		self.topics[0].append(root_topic)
		self.root_topic = root_topic

	def run(self):
		for i in range(self.iteration):
			logging.info('iteration: ' + str(i) + '\t processing: ' + str(len(self.corpus)) + ' documents')
			for document in self.corpus:
				self.process_document(document)
			self.print_count()
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
		roulette = np.cumsum(roulette)
		return bisect.bisect(np.cumsum(roulette), arrow)
		
	def print_topic(self, topic):
		string = ''
		if (topic.level == 1):
			string += '\n\n'
		if (topic.level == 2):
			string += '\n'
		string += ('level: ' + str(topic.level) + '\t')
		string += (str(topic.big_table_count) + '\t')
		string += (str(topic.small_table_count) + '\t')
		string += (self.top_words(topic.phi, 10) + '\n')
		return string

	def print_topics_recursively(self, topic, _string):
		string = self.print_topic(topic)
		for i in range(len(topic.children)):
			child = topic.children[i]
			string += self.print_topics_recursively(child, string)
		return string

	def print_state(self, i):
		logging.info('printing state\t' + self.param)
		if not os.path.isdir(exp_dir + self.param):
			os.mkdir(exp_dir + self.param)
		write_file = open(exp_dir + self.param + '/' + str(i) + '.txt', 'w')
		write_file.write(self.print_topics_recursively(self.root_topic, ''))
		write_file.close()

	def print_count(self):
		num_topics = np.zeros((self.depth_limit))
		num_tables = np.zeros((self.depth_limit))
		for i in range(self.depth_limit):
			num_topics[i] = len(self.topics[i])
			for topic in self.topics[i]:
				num_tables[i] += topic.small_table_count
		logging.info('num_topics: ' + str(num_topics))
		logging.info('num_tables: ' + str(num_tables))
		logging.info('num_average_tables 1: ' + str(num_tables / num_topics))
		logging.info('num_average_tables 2: ' + str(sum(num_tables) / sum(num_topics)))

	def process_document(self, document):
		if len(document.tables) == 0:
			random.shuffle(document.words)

		# table assignment
		for i in range(len(document.words)):
			word = document.words[i]

			# de-assignment
			old_table = document.word2table[i]

			# if first assignment, pass de-assignment
			if old_table == None:
				pass
			else:
				# remove previous assignment related to word
				old_table.words.remove(word)
				old_topic = old_table.topic
				old_topic.phi[word] -= 1
				old_topic.phi_sum -= 1

				# remove previous assignment of parents' related to word recursively
				parent_topic = old_topic.parent
				while(parent_topic != None):
					parent_topic.phi[word] -= 1
					parent_topic.phi_sum -= 1
					parent_topic = parent_topic.parent
				
				# if old_table has no word, remove it
				if len(old_table.words) == 0:
					document.tables.remove(old_table)
					old_topic.big_table_count -= 1
					old_topic.small_table_count -= 1
					parent_topic = old_topic.parent
					while(parent_topic != None):
						parent_topic.big_table_count -= 1
						parent_topic = parent_topic.parent
					
					# if old_topic, and their parents have no table assigned, remove them
					parent_topic = old_topic.parent
					if old_topic.big_table_count == 0:
						parent_topic.children.remove(old_topic)

					while(parent_topic != None):
						if parent_topic.big_table_count == 0:
							parent_topic.parent.children.remove(parent_topic)
							parent_topic = parent_topic.parent
						else:
							break

			# table assignment
			roulette = np.zeros((len(document.tables) + 1))
			for j in range(len(document.tables)):
				table = document.tables[j]
				roulette[j] = (table.topic.phi[word] / table.topic.phi_sum) * len(table.words)
			roulette[-1] = self.alpha / self.num_word
			new_table_index = self.choose(roulette)

			# error case
			if new_table_index == -1:
				print 'error 1'
				exit(-1)

			# create new table if last index is chosen
			if new_table_index == len(document.tables):
				new_table = Table()
				document.tables.append(new_table)
				new_topic = self.get_topic_for_table(new_table)
				new_table.topic = new_topic
				new_topic.big_table_count += 1
				new_topic.small_table_count += 1
				new_parent_topic = new_topic.parent
				while(new_parent_topic != None):
					new_parent_topic.big_table_count += 1
					new_parent_topic = new_parent_topic.parent
			else:
				new_table = document.tables[new_table_index]
				new_topic = new_table.topic
			new_table.words.append(word)
			new_topic.phi[word] += 1
			new_topic.phi_sum += 1
			new_parent_topic = new_topic.parent
			while(new_parent_topic != None):
				new_parent_topic.phi[word] += 1
				new_parent_topic.phi_sum += 1
				new_parent_topic = new_parent_topic.parent
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
			old_topic.big_table_count -= 1	
			old_topic.small_table_count -= 1									
			if old_topic.big_table_count == 0:
				parent_topic.children.remove(old_topic)

			while(parent_topic != None):
				for word in table.words:
					parent_topic.phi[word] -= 1					
				parent_topic.phi_sum -= len(table.words)
				parent_topic.big_table_count -= 1									
				if parent_topic.big_table_count == 0:
					parent_topic.parent.children.remove(parent_topic)
				parent_topic = parent_topic.parent

			new_topic = self.get_topic_for_table(table)
			table.topic = new_topic
			
			for word in table.words:
				new_topic.phi[word] += 1
			new_topic.phi_sum += len(table.words)
			new_topic.big_table_count += 1
			new_topic.small_table_count += 1			

			parent_topic = new_topic.parent
			while(parent_topic != None):
				for word in table.words:
					parent_topic.phi[word] += 1
				parent_topic.phi_sum += len(table.words)
				parent_topic.big_table_count += 1
				parent_topic = parent_topic.parent

	def get_topic_for_table(self, table):	
		# in some extreme case, if there are too many words in one table
		# we can have float overflow error, so for now we just limit the words in one table
		# that we use in sampling by this number. normally we don't have case where
		# words exceed 75 in one table, so it doesn't matter.
		# but still we should fix this later.
		word_limit = 75

		parent_topic = self.root_topic ## root_topic
		for i in range(1, self.depth_limit):
			topics = self.topics[i]
			roulette = np.zeros((len(topics) + 2))
			for j in range(len(topics)):
				topic = topics[j]
				roulette[j] =  topic.big_table_count
				for word in table.words[:word_limit]:
					roulette[j] *= (topic.phi[word] / topic.phi_sum * self.num_word)

			# index -1, create new child topic, and stop
			# index -2, choose current parent topic, and stop
			# otherwise, choose one of children and move down the tree
			roulette[-1] = self.gamma1 * math.pow(self.gamma2, i)
			roulette[-2] = parent_topic.small_table_count
			for word in table.words[:word_limit]:
				roulette[-2] *= (parent_topic.phi[word] / parent_topic.phi_sum * self.num_word)

			topic_index = self.choose(roulette)
			if topic_index == -1:
				logging.info('error in get_topic_for_table')
				logging.info('len(table.words):' + str(len(table.words)))
				exit(-1)
			if topic_index == len(topics) + 1:
				# create new topic
				topic = Topic(self.num_word, self.beta, parent_topic.level + 1)
				topic.parent = parent_topic
				parent_topic.children.append(topic)
				topic.phi = copy(topic.parent.phi) * self.delta
				topic.phi_sum = copy(topic.parent.phi_sum) * self.delta
				self.topics[i].append(topic)
				logging.info('created topic at level ' + str(parent_topic.level + 1))
				return topic
			if topic_index == len(topics):
				return parent_topic
			else:
				parent_topic = topics[topic_index]
		return parent_topic

def test_nips():
	alpha = 1.0
	beta = 0.1
	gamma1 = 0.1
	gamma2 = 1.5
	data_type = 'nips'
	iteration = 1000
	no_below = 5
	no_above = 0.2
	delta = 0.01
	depth_limit = 4
	model = Model(alpha, beta, gamma1, gamma2, data_type, iteration, no_below, no_above, delta, depth_limit)

	model.load_data()
	model.run()

test_nips()