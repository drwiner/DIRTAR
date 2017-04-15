#!/usr/bin/python

# modified the DIRT algorithm (Lin and Pantel, 2001, ACM)
# the slots are dependencies such as "nsubj", "dobj", and "pp-by", and many others, below

# Written by DAVID WINER
# Do not take without permission - drwiner@cs.utah.edu

from collections import namedtuple, Counter, defaultdict
from math import log2, sqrt
import operator
import pickle
from clockdeco import clock
import functools
from nltk.corpus import wordnet as wn

person = wn.synsets('person', wn.NOUN)[0]

def save_database(db, s_name):
	with open('dirtar_database_' + s_name + '.pkl', 'wb') as output:
		pickle.dump(db, output, protocol=pickle.HIGHEST_PROTOCOL)

# NA
# EXCLUDE = 'is are be was were said have has had and or >comma >squote >rparen >lparen >period >minus >ampersand'.split()

LEFT_DEPS = ['nsubj', 'nsubj:xsubj', 'nsubjpass', 'nmod:poss']
REVERSIBLE_LEFTS = ['nsubjpass']
RIGHT_DEPS = ['iobj, dobj', 'nmod:at', 'nmod:from', 'nmod:by', 'nmod:to', 'nmod:agent', 'nmod:in', 'nmod:into', 'nmod:poss', 'nmod:through', 'nmod:on', 'nmod:across', 'nmod:over', 'nmod:away_from']
REVERSIBLE_RIGHTS = ['nmod:agent', 'nmod:by']
MULTI_SLOTS = LEFT_DEPS + RIGHT_DEPS

Triple = namedtuple('Triple', ['X', 'path', 'Y'])

def cleanLine(line):
	return ' '.join(line.split()) + ' '


def wordnet_replace(word, ner):
	synsets = wn.synsets(word, wn.NOUN)
	if len(synsets) == 0 or ner == 'PERSON':
		return person.lemma_names()[0]
	h_paths = synsets[0].hypernym_paths()[0]
	if len(h_paths) < 6:
		return h_paths[-1].lemma_names()[0]
	return h_paths[5].lemma_names()[0]

@clock
def readCorpus(clause_file):
	with open(clause_file, 'r') as clauses:
		for i, line in enumerate(clauses):
			# print(i)
			entries = line.split(',')
			if len(entries) != 3:
				# there's a comma or number as a word
				if entries[0] == '(':
					temp = [None, None, None]
					temp[0] = '(comma {}'.format(entries[1])
					temp[1] = entries[2]
					temp[2] = entries[3]
					entries = temp
				else:
					temp = [None, None, None]
					temp[0] = entries[0]
					temp[1] = entries[1]
					temp[2] = '(number - NP - NUMBER - dobj)'
					entries = temp
			if len(entries) != 3:
				print(line)
				continue
			# entry 0 and 2 are (X_orth - X_pos - X_ner - X_dep)
			x_pieces = entries[0].split(' - ')
			y_pieces = entries[2].split(' - ')
			x = x_pieces[0].split('(')[1].strip().lower()
			y = y_pieces[0].split('(')[1].strip().lower()
			path = entries[1].strip()

			# Straightforward X and Y
			TStream.append(Triple(x, path, y))

			# X and Y separated by dependencies, can filter by legal dependencies later
			x_dep = x_pieces[-1].split(')')[0].strip()
			x_ner = x_pieces[2].strip()

			y_dep = y_pieces[-1].split(')')[0].strip()
			y_ner = y_pieces[2].strip()

			# Wstream : swap x and y with wordnet
			wn_x = wordnet_replace(x, x_ner)
			wn_y = wordnet_replace(y, y_ner)
			WStream.append(Triple(wn_x, path, wn_y))

			# left and right are tuples (noun, dep, ner)
			MStream.append(Triple((x, x_dep, x_ner), path, (y, y_dep, y_ner)))

			# X and Y collapsed but filtered by dependency type
			if x_dep in LEFT_DEPS and y_dep in RIGHT_DEPS:
				FTStream.append(Triple(x, entries[1], y))
				# X and Y recollapsed and filtered by dependency type
				x_prime, y_prime = decide_swap(x, y, x_dep, y_dep)
				FCTStream.append(Triple(x_prime, path, y_prime))

			# X and Y recollapsed by dependency type
			x_prime, y_prime = decide_swap(x, y, x_dep, y_dep)
			CTStream.append(Triple(x_prime, path, y_prime))



def decide_swap(x, y, x_dep, y_dep):
	if x_dep != 'none':
		if x_dep  in REVERSIBLE_LEFTS:
			# swap
			return y, x
	if y_dep != 'none':
		if y_dep in REVERSIBLE_RIGHTS:
			# swap
			return y, x
	# no swap
	return x, y

@clock
def apply_MinfreqFilter(stream, stream_name, min_freq):

	PCounter = Counter([t.path for t in stream])
	distinct_unfiltered_Pinstances = set(PCounter.keys())
	distinct_filtered_Pinstances = set(p for p in distinct_unfiltered_Pinstances if PCounter[p] >= min_freq)
	filtered_Tinstances = [t for t in stream if PCounter[t.path] >= min_freq]

	print('{}\n'.format(stream_name))
	print('Found {} distinct paths, {} after minfreq filtering'.format(len(distinct_unfiltered_Pinstances), len(distinct_filtered_Pinstances)))
	print('Found {} path instances, {} after minfreq filtering'.format(len(stream), len(filtered_Tinstances)))

	return len(distinct_unfiltered_Pinstances), len(distinct_filtered_Pinstances), len(stream), len(filtered_Tinstances), filtered_Tinstances

# Entry = namedtuple('Entry', ['word', 'count'=0, 'mi'])
class Entry:
	def __init__(self, path=None, slot=None, word=None, dep=None, ner=None):
		self.path = path
		self.slot = slot
		self.word = word
		self.count = 1
		self.mi = None
		if dep is not None:
			self.dep = dep
		if ner is not None:
			self.ner = ner

	def __hash__(self):
		return hash(self.path ^ self.slot ^ self.word)

	def update(self, path, slot, word):
		if self.word is None:
			self.word = word
		if self.slot is None:
			self.slot = slot
		if self.path is None:
			self.path = path
		self.count += 1

	def __repr__(self):
		return '{}\t{}\t{}\t{}'.format(self.path, self.slot, self.word, self.mi)


@clock
def loadDatabase(db, filtered_Tinstances):

	# For each filtered triple instance:
	for x, path, y in filtered_Tinstances:
		if path is None or path == 'none':
			continue
		## Triple Database by-path (default dict, so creates dict at key [path] or reference existing)
		path_db = db[path]

		if x is not None and str(x) != 'none':
			if 'X' not in path_db.keys():
				path_db['X'] = dict()
			pdbx = path_db['X']

			if x in pdbx.keys():
				pdbx[x].count += 1
			else:
				pdbx[x] = Entry(path, 'X', x)

		if y is not None and str(y) != 'none':
			if 'Y' not in path_db.keys():
				path_db['Y'] = dict()
			pdby = path_db['Y']

			if y in pdby.keys():
				pdby[y].count += 1
			else:
				pdby[y] = Entry(path, 'Y', y)


# @clock
def MI(db, wsc, sc, path, slot_pos, word):
	# db - which database
	# wsc - word slot count
	# sc - slot count

	pdb = db[path][slot_pos]

	# |p,s,w|
	psw = pdb[word].count
	if psw == 0:
		return 0

	# |p, s, *|
	ps_ = sum(pdb[word].count for word in pdb.keys())
	if ps_ == 0:
		return 0

	# |*, s, w|
	_sw = 0
	if word in wsc.keys() :
		_sw = wsc[word][slot_pos]
	if _sw == 0:
		return 0

	# |*, s, *|
	_s_ = sc[slot_pos]

	if (psw * _s_) < 0:
		return 0
	if (ps_ * _sw) < 0:
		return 0
	mi = log2((psw * _s_) / (ps_ * _sw))
	if mi < 0:
		return 0
	return mi


@clock
def loadDatabase_multislot(db, filtered_Tinstances):

	# For each filtered triple instance:
	for x, path, y in filtered_Tinstances:
		if path is None or path == 'none':
			continue
		## Triple Database by-path (default dict, so creates dict at key [path] or reference existing)
		path_db = db[path]
		x_noun, x_dep, x_ner = x
		y_noun, y_dep, y_ner = y
		x_slot_name = 'x_' + x_dep
		y_slot_name = 'y_' + y_dep

		if x_noun is not None and x != 'none':
			if x_slot_name not in path_db.keys():
				path_db[x_slot_name] = dict()
			pdbx = path_db[x_slot_name]
			if x_noun in pdbx.keys():
				pdbx[x_noun].count += 1
			else:
				pdbx[x_noun] = Entry(path, x_slot_name, x_noun, dep=x_dep, ner=x_ner)

		if y_noun is not None and y_noun != 'none':
			if y_slot_name not in path_db.keys():
				path_db[y_slot_name] = dict()
			pdby = path_db[y_slot_name]
			if y in pdby.keys():
				pdby[y_noun].count += 1
			else:
				pdby[y_noun] = Entry(path, y_slot_name, y_noun, dep=y_dep, ner=y_ner)


@clock
def updateMI(db, word_slot_count, slot_count):
	# db = databases[i]
	# word_slot_count = word_slot_counts[i]
	# slot_count = slot_counts[i]
	_x_ = 0
	_y_ = 0
	for p in db.keys():
		for w in db[p]['X'].keys():
			x = db[p]['X'][w].count
			_x_ += x
			if w not in word_slot_count.keys():
				word_slot_count[w] = dict()
				word_slot_count[w]['Y'] = 0
				word_slot_count[w]['X'] = 0
			word_slot_count[w]['X'] += x

		for w in db[p]['Y'].keys():
			y = db[p]['Y'][w].count
			_y_ += y
			if w not in word_slot_count.keys():
				word_slot_count[w] = dict()
				word_slot_count[w]['Y'] = 0
				word_slot_count[w]['X'] = 0
			word_slot_count[w]['Y'] += y
	slot_count['X'] = _x_
	slot_count['Y'] = _y_

	for path in db.keys():

		tdpx = db[path]['X']
		tdpy = db[path]['Y']

		for entry in tdpx.values():
			entry.mi = MI(db, word_slot_count, slot_count, entry.path, entry.slot, entry.word)
		for entry in tdpy.values():
			entry.mi = MI(db, word_slot_count, slot_count, entry.path, entry.slot, entry.word)

def updateMI_multislot(db, word_slot_count, slot_count):
	for p in db.keys():
		slots = db[p].keys()
		for slot in slots:
			for w in db[p][slot].keys():
				s = db[p][slot][w].count
				slot_count[slot] += s
				if w not in word_slot_count.keys():
					word_slot_count[w] = defaultdict(int)
				word_slot_count[w][slot] += s

	for path in db.keys():
		# slot, entries
		for slot_name, nouns in db[path].items():
			tdps = db[path][slot_name]
			for entry in tdps.values():
				if entry.word == 'none':
					entry.mi = 0
					continue
				if type(tdps[entry.word]) != Entry:
					entry.mi = 0
					continue
				if type(tdps[entry.word]) == defaultdict:
					entry.mi = 0
				if entry.word not in tdps.keys():
					entry.mi = 0
					continue
				entry.mi = MI(db, word_slot_count, slot_count, entry.path, entry.slot, entry.word)

# @clock
def pathSim(p1, p2):
	slot_x_sim = slotSim(p1, p2, 'X')
	slot_y_sim = slotSim(p1, p2, 'Y')

	return sqrt(slot_x_sim * slot_y_sim)


def pathSimdb(p1, p2, db):
	slot_x_sim = slotSimdb(p1, p2, 'X', db)
	slot_y_sim = slotSimdb(p1, p2, 'Y', db)

	return sqrt(slot_x_sim * slot_y_sim)

def pathSim_multiSlot(p1, p2, db):
	slots = set(db[p1].keys()) & set(db[p2].keys())
	sim_list = [slotSimdb(p1, p2, slot, db) for slot in slots]
	k = len(sim_list)
	if k == 0:
		return 0
	a = functools.reduce(operator.mul, sim_list, 1)
	return a**(1./float(k))


def weighted_pathSim_multiSlot(p1, p2, db):
	slots = set(db[p1].keys()) & set(db[p2].keys())
	weights = [len(db[p1][slot]) + len(db[p2][slot]) / 2 for slot in slots]
	slot_weight = zip(slots, weights)
	sim_list = [w*slotSimdb(p1, p2, slot, db) for slot, w in slot_weight]
	k = len(sim_list)
	if k == 0:
		return 0
	a = functools.reduce(operator.mul, sim_list, 1)
	return a ** (1. / float(k))


def slotSimdb(p1, p2, slot_pos, db):
	wd1 = db[p1][slot_pos]
	wd2 = db[p2][slot_pos]

	pd1 = set(wd1.keys())
	pd2 = set(wd2.keys())

	n_score = 0
	for word in pd1.intersection(pd2):
		n_score += wd1[word].mi + wd2[word].mi
	if n_score == 0:
		return 0

	d_score_1, d_score_2 = 0, 0
	for word in pd1:
		d_score_1 += wd1[word].mi
	for word in pd2:
		d_score_2 += wd2[word].mi
	d_score = d_score_1 + d_score_2
	if d_score == 0:
		return 0

	return n_score / d_score


# @clock
def slotSim(p1, p2, slot_pos):
	wd1 = triple_database[p1][slot_pos]
	wd2 = triple_database[p2][slot_pos]

	pd1 = set(wd1.keys())
	pd2 = set(wd2.keys())

	n_score = 0
	for word in pd1.intersection(pd2):
		n_score += wd1[word].mi + wd2[word].mi
	if n_score == 0:
		return 0

	d_score_1, d_score_2 = 0, 0
	for word in pd1:
		d_score_1 += wd1[word].mi
	for word in pd2:
		d_score_2 += wd2[word].mi
	d_score = d_score_1 + d_score_2
	if d_score == 0:
		return 0

	return n_score/d_score


def most_similar_to(test_lemma, db):

	if test_lemma == ' ' or test_lemma == '\n' or test_lemma not in db.keys():
		return None

	path_test = dict()
	for p in db.keys():
		ps = pathSimdb(test_lemma, p, db)
		path_test[p] = ps

	return list(reversed(sorted(path_test.items(), key=operator.itemgetter(1))))


import semantic_parser

def most_similar_with_multislot_with_semantic(test_lemma, db, semantic=1):

	if test_lemma == ' ' or test_lemma == '\n' or test_lemma not in db.keys():
		return None
	if semantic:
		tp_lemma = semantic_parser.filter_action_lemma(test_lemma, db)

	reg_test = dict()
	sem_test = dict()
	weighted_reg_test = dict()
	weighted_sem_test = dict()

	for p in set(db.keys()):
		print(p)

		reg_test[p] = pathSim_multiSlot(test_lemma, p, db)
		weighted_reg_test[p] = weighted_pathSim_multiSlot(test_lemma, p, db)
		if semantic:
			print(tp_lemma)
		#tests with tp_lemma
		if semantic:
			if tp_lemma is None:
				sem_test[p] = 0
				weighted_sem_test[p] = 0
			else:
				sem_test[p] = pathSim_multiSlot(tp_lemma, p, db)
				weighted_sem_test[p] = weighted_pathSim_multiSlot(tp_lemma, p, db)

	if semantic:
		g_semantic = list(reversed(sorted(sem_test.items(), key=operator.itemgetter(1))))
		w_semantic = list(reversed(sorted(weighted_sem_test.items(), key=operator.itemgetter(1))))
	g_regular = list(reversed(sorted(reg_test.items(), key=operator.itemgetter(1))))
	w_regular = list(reversed(sorted(weighted_reg_test.items(), key=operator.itemgetter(1))))

	if semantic:
		return (g_regular, g_semantic, w_regular, w_semantic)
	return (g_regular, w_regular)


# run both geo and weighted-geo measures
def most_similar_to_test(test_paths, db):
	tested_dict = dict()
	for tp in test_paths:
		if tp == ' ' or tp == '\n' or tp not in db.keys():
			continue

		test_paths = dict()
		for p in db.keys():
			test_paths[p] = pathSimdb(tp, p, db)

		tested_dict[tp] = list(reversed(sorted(test_paths.items(), key=operator.itemgetter(1))))

	return tested_dict


def test_most_similar_to(i, action_lemma_doc, k_most_similar, line1, line2):
	db = databases[i]
	test_d = most_similar_to(action_lemma_doc, db)

	with open('movie_output-' + str(i), 'w') as ot:
		ot.write(line1)
		ot.write(line2)
		for lemma, paths in test_d.items():
			ot.write('\n')
			ot.write('MOST SIMILAR RULES FOR: "{}"\n'.format(lemma))
			if lemma not in db.keys():
				ot.write('This phrase is not in the triple database.\n')
			else:
				for i in paths:
					if i >= k_most_similar:
						if paths[i] != paths[-1]:
							break
					p, score = paths[i]
					if score > 0:
						ot.write(str(i+1) + '. \"%s\" %24.12f\n' % (p, score))
					else:
						break

# http://stackoverflow.com/questions/19189274/defaultdict-of-defaultdict-nested
def rec_dd():
	return defaultdict(rec_dd)

import sys

if __name__ == '__main__':
	for arg in sys.argv:
		print(arg, end=' ')
	print('\n')

	if len(sys.argv) != 4 and len(sys.argv) > 1:
		print('must have 3 args')
		raise AssertionError
	elif len(sys.argv) == 1:
		# for testing
		min_freq = 5
		corpus_text = 'movie_clauses.txt'
		test_text = 'action_lemmas.txt'
	else:
		min_freq = int(sys.argv[-1])
		corpus_text = str(sys.argv[1])
		test_text = str(sys.argv[2])


	# TStream (triple stream) - collects all triple instances
	TStream = []
	# v_dep_dict (verb dependency dict) keys are (verb, dependency, slot) tuples values are noun lists
	MStream = []

	# CTStream (collapsed triple stream) a list of triple instances, sometimes with x and y flipped
	CTStream = []
	# FTSream whose instances are filtered by whether any of the X or Y of that triple aren't legal
	FTStream = []
	# FCTStream whose instances are filtere (as above), sometimes with x and y flipped
	FCTStream = []

	# replace words with wordnet level 6 or leave
	WStream = []

	print('reading and sorting input clauses')
	#### READ CLAUSES ####
	readCorpus(corpus_text)
	######################

	# stats:
	# print('writing x dep values')
	# with open('xdeps.txt', 'w') as xdeps:
	# 	for key, value_list in v_x_dep_dict.items():
	# 		xdeps.write(key + '\n')
	# 		for val in value_list:
	# 			xdeps.write(val + '\n')
	# 		xdeps.write('\n')
	# print('writing y dep values')
	# with open('ydeps.txt', 'w') as ydeps:
	# 	for key, value_list in v_y_dep_dict.items():
	# 		ydeps.write(key + '\n')
	# 		for val in value_list:
	# 			ydeps.write(val + '\n')
	# 		ydeps.write('\n')


	streams = [TStream, CTStream, FTStream, FCTStream, WStream]
	s_names = ['tstream', 'ctstream', 'ftstream', 'fctstream', 'wstream']

	# triple_databases - Triple Database - collection of triple instances by path
	triple_database = rec_dd()
	# FStream
	triple_dep_filtered = rec_dd()
	# CTStream
	triple_collapsed = rec_dd()
	# FCTStream
	triple_dep_filtered_collapsed = rec_dd()
	# WStream
	triple_W_db = rec_dd()


	databases = [triple_database, triple_dep_filtered, triple_collapsed, triple_dep_filtered_collapsed, triple_W_db]

	ftinstances = []
	slot_counts = []
	word_slot_counts = []
	for i in range(len(streams)):
		ftinstances.append(list())
		slot_counts.append(dict())
		word_slot_counts.append(dict())


	for i in range(len(streams)):
		print(i)
		print('applying filter')
		#### Apply MinFreq####
		dp, dmf, pi, pimf, ftinstances = apply_MinfreqFilter(streams[i], s_names[i], min_freq)
		# save meta path info
		# output_paths_info[i] = (dp, dmf, pi, pimf)
		# load database
		print('loading database')
		loadDatabase(databases[i], ftinstances)

		# apply semantic discrimination for each action lemma if in database
		# use test doc to filter by semantic, but for now, just update
		print('updating MI')
		updateMI(databases[i], word_slot_counts[i], slot_counts[i])

		print('after MI, dumping database')
		save_database(databases[i], s_names[i])

		line1 = 'Found {} distinct paths, {} after minfreq filtering.\n'.format(dp, dmf)
		line2 = 'Found {} path instances, {} after minfreq filtering.\n'.format(pi, pimf)



	# MStream
	triple_multi_slot = rec_dd()
	dp, dmf, pi, pimf, ftinstances = apply_MinfreqFilter(MStream, 'mstream', min_freq)
	loadDatabase_multislot(triple_multi_slot, ftinstances)
	updateMI_multislot(triple_multi_slot, dict(), defaultdict(int))
	save_database(triple_multi_slot, 'mstream')