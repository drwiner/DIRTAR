# Using Stanford Parser to piece together clauses
import collections
class Word:
	def __init__(self, token, dep_tuple):
		self.originalText = token['originalText']
		self.pos = token['pos']
		self.ner = token['ner']
		self.lemma = token['lemma']
		self.dep = dep_tuple[0]
		self.parent = dep_tuple[1]

# class Clause:
# 	def __init__(self, verb):
# 		self.verb = verb
# 		self.nsubj = None

from collections import namedtuple
Clause = namedtuple('Clause', 'verb nsubj dobj pp_by'.split())

class Sentence:
	"""
	A class which acts like a list of words. the words are sentences.
	The instance should have values for verbphrase, nsubj, dobj, and pp_by
	"""

	def __init__(self, nlp_sent):
		"""
			:param nlp_sent: sentence extracted from parse from stanford corenlp parser
			has "enhancedPlusPlusDependencies"
			has "tokens"
		"""
		tokens = nlp_sent['tokens']
		deps = nlp_sent['enhancedPlusPlusDependencies']
		dep_dict = {dep['dependentGloss']: (dep['dep'], dep['governorGloss']) for dep in deps}

		# create sentence list __self__
		self.word_list = self.make_words(tokens, dep_dict)

		# initialize slot_dict
		self.slot_dict = collections.defaultdict(set)

		# populate slot_dict
		self.parse_to_slot(self.slot_dict)

	def __len__(self):
		return len(self.word_list)

	def __getitem__(self, index):
		return self.word_list[index]

	def make_words(self, tokens, dep_dict):
		return [Word(token, dep_dict[token['word']]) for token in tokens]

	def parse_to_slots(self, slot_dict):
		for word in self.word_list:
			if word.dep == 'ROOT':
				slot_dict['verb'].add(word)
				continue
			if word.dep == 'conj:and':
				if word.parent in slot_dict[word]:
					slot_dict[word.dep].add(word)
				continue
			if word.dep in {'nsubj', 'nsubjpass', 'dobj', 'nmod:at', 'nmod:to'}:
				slot_dict[word.dep].add(word)
				continue
			if word.lemma == 'by' and word.dep == 'case':
				slot_dict['pp_by'].add(word.parent)

		if len(slot_dict['verb']) == 0:
			raise ValueError('no ROOT in this sentence:{}'.format(self.word_list))

	def __repr__(self):
		return ' '.join(word.originalText for word in self.word_list)