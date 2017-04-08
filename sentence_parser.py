# Using Stanford Parser to piece together clauses

class Word:
	def __init__(self, token, dep_tuple):
		self.originalText = token['originalText']
		self.pos = token['pos']
		self.ner = token['ner']
		self.lemma = token['lemma']
		self.dep = dep_tuple[0]
		self.parent = dep_tuple[1]

class Clause:
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
		self.Word_List = self.make_words(tokens, dep_dict)

		self.verb_phrase = None
		self.nsubj = None
		self.dobj = None
		self.pp_by = None

	def __len__(self):
		return len(self.Word_List)

	def __getitem__(self, index):
		return self.Word_List[index]

	def make_words(self, tokens, dep_dict):
		return [Word(token, dep_dict[token['word']]) for token in tokens]

	def find_verb(self):
		found = False
		for word in self.Word_List:
			if word.dep == 'ROOT':
				self.verb_phrase = word.lemma
				found = True
		if not found:
			raise ValueError('no ROOT in this sentence:{}'.format(self.Word_List))

	def find_nsubj(self):
		for word in self.Word_List:
			# if word.dep
			pass
		pass

	def find_dobj(self):
		pass
	def find_pp_by(self):
		pass

	def __repr__(self):
		return ' '.join(word.originalText for word in self.Word_List)