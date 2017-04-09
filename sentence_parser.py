# Using Stanford Parser to piece together clauses

# import collections
from nltk.tree import ParentedTree
# from nltk.tree import Tree

class Word:
	# pretends to be token + tree
	def __init__(self, token, dep_tuple):
		self.originalText = token['originalText']
		self.pos = token['pos']
		self.ner = token['ner']
		self.lemma = token['lemma']
		self.dep = dep_tuple[0]
		self.parent = dep_tuple[1]
	def __repr__(self):
		return '(Word: ' + self.originalText + ')'

def get_left_sibling(subtree):
	current = subtree
	while current.parent() is not None:
		while current.left_sibling() is not None:
			if current.left_sibling().label() == "NP":
				return current.left_sibling()
			current = current.left_sibling()
		current = current.parent()
	return None

def get_right_sibling(subtree):
	current = subtree
	while current.parent() is not None:
		while current.right_sibling() is not None:
			if current.right_sibling().label() == "NP":
				return current.right_sibling()
			if current.right_sibling().label() == 'PP':
				if current.right_sibling()[0].leaves()[0] == 'by':
					return current.parent()[1]
			# elif current.right_sibling().label() == 'VP' and current.right_sibling()[0][0] == 'by':
			# 	return current.right_sibling()
			current = current.right_sibling()
		current = current.parent()
	return None

def parse_to_clauses(ptree):
	clauses = []
	for tree in ptree.subtrees(filter=lambda x: x.label() in {'VBZ', 'VBN'}):
		if tree.leaves()[0] in {'is', 'be', 'have', 'was', 'has', 'had', 'of', 'would', 'were', 'are'}:
			continue

		clause_tuple = (get_left_sibling(tree), tree, get_right_sibling(tree))
		clauses.append(clause_tuple)
	return clauses

def head_noun(ptree):
	if ptree is None:
		return None
	if len(ptree) == 1:
		return ptree.leaves()[0]
	if len(ptree) > 1:
		# try:
		sub_list = list(ptree.subtrees(filter=lambda x: x.label() in {'NNS', 'NNP', 'NN'}))
		if len(sub_list) > 0:
			sub = sub_list[0]
			return sub.leaves()[0]

		return None

from collections import namedtuple
# fake_token = namedtuple('fake_token', 'lemma'.split())
Clause_relation = namedtuple('Clause_relation', 'verb dep relation'.split())

class Sentence:
	"""
	Reads sentence, spits clauses
	"""
	none_tuple = ('None', 'None', 'None', 'None')

	def __init__(self, nlp_sent):
		"""
			:param nlp_sent: sentence extracted from parse from stanford corenlp parser
			has "enhancedPlusPlusDependencies"
			has "tokens"
		"""
		tokens = nlp_sent['tokens']
		self.raw_dict = {token['word']: token for token in tokens}
		const_parse = ParentedTree.fromstring(nlp_sent['parse'])
		self.clause_trees = parse_to_clauses(const_parse)
		# dependencies
		deps = nlp_sent['enhancedPlusPlusDependencies']
		dep_dict = {dep['dependentGloss']: (dep['dep'], dep['governorGloss']) for dep in deps}

		# create sentence list __self__
		self.word_list = self.make_words(tokens, dep_dict)
		self.word_dict = dict(zip(self.raw_dict.keys(), self.word_list))

		self.clauses = self.integrate_tokens_to_clauses()

		# self.clause_relations = self.build_clause_relations(self.slot_dict)

	def __len__(self):
		return len(self.word_list)

	def __getitem__(self, index):
		return self.word_list[index]

	def make_words(self, tokens, dep_dict):
		return [Word(token, dep_dict[token['word']]) for token in tokens]

	def integrate_tokens_to_clauses(self):
		clauses = []
		for tree in self.clause_trees:
			head_left = head_noun(tree[0])
			head_right = head_noun(tree[2])
			verb = tree[1].leaves()[0]
			if head_left is not None:
				word = self.word_dict[head_left]
				left_thing = (word.lemma, tree[0].label(), word.ner, word.dep)
				if word.dep == 'punct':
					left_thing = self.none_tuple
			else:
				left_thing = self.none_tuple
			if head_right is not None:
				word = self.word_dict[head_right]
				right_thing = (word.lemma, tree[2].label(), word.ner, word.dep)
				if word.dep == 'punct':
					right_thing = self.none_tuple
			else:
				right_thing = self.none_tuple
			verb_lemma = self.raw_dict[verb]['lemma']
			if left_thing[0] == 'None' and right_thing[0] == 'None':
				continue
			if verb_lemma == 'be':
				continue
			clauses.append((left_thing, verb_lemma, right_thing))
		return clauses

	def __repr__(self):
		return ' '.join(word.originalText for word in self.word_list)


def assemble_clause_relations(doc_sents):
	return [clause for sent in doc_sents for clause in sent.clauses]

def read_corpus(file_name):
	print('reading')
	raw_doc = ''
	with open(file_name) as fn:
		# raw_doc += ' '.join(fn.readlines())
		for line in fn:
			sp = line.split()
			raw_doc += ' '.join(wrd.strip() for wrd in sp if wrd != '\n')
			raw_doc += ' '
	print('finished reading, now parsing nlp style')
	return nlp(raw_doc)

if __name__ == '__main__':
	from pycorenlp import StanfordCoreNLP
	from functools import partial

	nlp_server = StanfordCoreNLP('http://localhost:9000')
	nlp = partial(nlp_server.annotate, properties={'outputFormat': 'json'})
	# test_doc = nlp('He licks his lips nervously, squeezes his eyes shut, and hits the button.')
	# nlp_sent = test_doc['sentences'][0]
	# clauses = parse_to_clauses(ParentedTree.fromstring(nlp_sent['parse']))
	# test_doc = nlp('They walk forward slowly, carrying their helmets, up the ramp and into the tunnel of light, following the Martian, who retreats before them.')
	SAVED = True
	if not SAVED:
		parsed_docs = read_corpus('random_test.txt')
	else:
		import pickle
		parsed_docs = pickle.load(open('random_test_dump', 'rb'))
	doc_sents = [Sentence(sent) for sent in parsed_docs['sentences']]
	clauses = assemble_clause_relations(doc_sents)

	with open('clauses.txt', 'w') as clause_file:
		for clause in clauses:
			clause_file.write('({} - {} - {} - {}),\t{},\t({} - {} - {} - {})\n'.format(clause[0][0], clause[0][1], clause[0][2], clause[0][3],
			                                                                     clause[1],
			                                                                     clause[2][0], clause[2][1], clause[2][2], clause[2][3]))

