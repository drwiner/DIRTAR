#!/usr/bin/python

import pickle
import spacy
import format_corpus as fcorp
import dirtar as mdirt
from dirtar import Entry

def cleanLine(line):
	return ' '.join(line.split()) + ' '


def parse_clause(sent):
	parse_list = []
	s_doc = nlp(sent)
	nchunks = list(s_doc.noun_chunks)
	nchunk_indices, nchunk_dict = fcorp.noun_chunk_index_list(nchunks)
	first_noun_encountered = False
	for i, token in enumerate(s_doc):
		# ignore noun phrases, but keep track of first and last
		if i in nchunk_indices:
			if i in nchunk_dict.keys():
				if first_noun_encountered:
					break
				first_noun_encountered = True
			# dont' add nouns to parse_list
			continue
		if first_noun_encountered and token.orth_ in fcorp.symb_dict.keys():
			parse_list.append(fcorp.symb_dict[token.orth_])
		elif first_noun_encountered:
			parse_list.append(token.orth_)
		if len(parse_list) > 0 and parse_list[-1] == '>RPAREN':
			parse_list = []
	return ' '.join(parse_list) + ' '


def clause_to_path(clause):
	if clause[0].isupper():
		# then read this clause as is
		path = parse_clause(clause)
	else:
		# add a pronoun just to get the parsing right...
		path = parse_clause('he' + clause)

	return path


def raw_sent_to_path_list(raw_sent):
	path_list = []
	clauses = raw_sent.split(',')
	path_list.append(parse_clause(clauses[0].strip()))
	if len(clauses) > 1:
		for c in clauses[1:]:
			if c.isspace() or c == '\n':
				continue
			# intermediate stage "clause_to_path" makes sure we get parsing right
			path_list.append(clause_to_path(c.strip()))
	return path_list


def parse_scene_sents(file_name):
	# each entry in this list is a tuple of the form (clause, action_list)
	score_sent_list = []
	with open(file_name) as sc_sents:
		for line in sc_sents:
			raw_sent, raw_actions = line.split('-#-')
			actions = raw_actions.split()
			paths = raw_sent_to_path_list(raw_sent)
			score_sent_list.append((paths, [act for act in actions if act in ACTION_TYPES]))
			# score_sent_list.extend([(path, actions) for path in paths])
	return score_sent_list


def parse_key_paths(file_name):
	key_dict = dict()
	new_key = False
	current_key = None
	with open(file_name) as key_phrases:
		for line in key_phrases:
			if line == '-\n':
				#new key, on next line
				new_key = True
			elif new_key:
				new_key = False
				# turn the phrase such as "look at" into "look-at"
				current_key = '-'.join(line.split())
				ACTION_TYPES.append(current_key)
				key_dict[current_key] = set()
				key_dict[current_key].add(cleanLine(line))
			else:
				key_dict[current_key].add(cleanLine(line))
	return key_dict


def best_phrase_score_tuple_list_intersection(test_phrase, list1, list2, db):
	# list1 are k-most similar phrases to "test phrase"
	# list2 are k-most similar phrases to cndt "key phrase"
	"""
	for item in intersection:
		find phrase most similar to original (to list1[0][0])
	similarity of test phrase to similar phrase, times similarity of
	"""
	set1 = {phrase for phrase, score in list1}
	set2 = {phrase for phrase, score in list2}
	set_cap = set1.intersection(set2)
	if len(set_cap) == 0:
		return None, 0
	elif len(set_cap) > 1:
		best_phrase = None
		max_sc = 0
		for phrase in set_cap:
			sc = mdirt.pathSimdb(test_phrase, phrase, db)
			if sc > max_sc:
				max_sc = sc
				best_phrase = phrase
		return best_phrase, max_sc
	phrase = set_cap.pop()
	return phrase, mdirt.pathSimdb(test_phrase, phrase, db)


def collect_assignments(key_phrases, test_list, db, k):
	"""
	For each test item in test_list, make assignment and score whether correct, and how many missed, etc.

	:param key_phrases: keys are actions whose value is a set of phrases for that key
	:param test_list: each item is a tuple of the form (phrase_list, action_list) from duel corpus
	:param db: triple database
	:param k: k-most similar paths considered (most similar to test-item phrase)
	:return: precision, recall, and fscore for each test_item
	"""

	# mdirt_dict: each key is a key phrase and its value is a ranked list of (phrase, score) tuples
	mdirt_dict = dict()
	for ky, phrases in key_phrases.items():
		mdirt_dict.update(mdirt.most_similar_to(phrases, db))
	# mdirt_dict = mdirt.most_similar_to(key_phrases.values(), db)
	assignments = []
	for phrase_list, action_list in test_list:
		# action_phrase_dict = {act: key_phrases[act] for act in action_list}
		# assignments.append([])
		action_checklist = set(action_list)
		unlabeled_phrases = 0
		for phrase in phrase_list:

			if phrase not in mdirt_dict.keys():
				unlabeled_phrases += 1
				continue

			# get k most similar to phrase
			ranked_dict = mdirt.most_similar_to([phrase], db)
			k_ranked_list = ranked_dict[phrase][:k]

			# for key key phrase, check
			best_keyphrase = None
			best_score = 0

			key_phrase_list = mdirt_dict.keys()
			for kp in key_phrase_list:
				if kp == phrase:
					best_keyphrase = kp
			if best_keyphrase is None:
				# find best keyphrase if possible.
				for key_phrase, ranked_list in mdirt_dict.items():
					best_phrase, sc = best_phrase_score_tuple_list_intersection(phrase, k_ranked_list, ranked_list[:k], db)
					if best_phrase is None:
						continue

					if best_phrase == phrase:
						best_keyphrase = key_phrase
						break

					if sc > best_score:
						best_score = sc
						best_keyphrase = key_phrase

			if best_keyphrase is None:
				unlabeled_phrases += 1
				# print('{}\t{}\n'.format(phrase, action_list))
				# assignments.append((phrase, None, None, action_list))
				continue

			for key, k_phrases in key_phrases.items():
				for keyphrase in k_phrases:
					if best_keyphrase == keyphrase:
						if key in action_checklist:
							action_checklist.remove(key)
						print('{}\t{}\t{}\t{}\n'.format(phrase, best_keyphrase, key, action_list))

						if key in action_list:
							assignments.append((phrase, best_keyphrase, key, action_list, str(1)))
						else:
							assignments.append((phrase, best_keyphrase, key, action_list, str(0)))

		if unlabeled_phrases > 0:
			for action in action_checklist:
				score = str(unlabeled_phrases / len(action_checklist))
				print('unlabeled_phrases_found: {}\t{}\t{}\n'.format(unlabeled_phrases, action, score))
				assignments.append(('unlabeled', unlabeled_phrases, action, score))

	return assignments


if __name__ == '__main__':
	print('reading spacy english')
	nlp = spacy.load('en')
	print('finished reading spacy english')

	# import key paths (action-paths)
	ACTION_TYPES = []
	print('importing key paths')
	key_phrase_dict = parse_key_paths('key_phrases')
	print('finished importing key paths')

	# import test sentences
	print('parsing sentence key')
	# each item in the list is a tuple (phrase_list, action_list)
	score_list = parse_scene_sents('IE_sent_key.txt')
	print('finished parsing sentence key')

	# import triple database
	print('loading triple database for scoring')
	with open('trip_database.pkl', 'rb') as tripdatabase:
		trip = pickle.load(tripdatabase)
	print('finished loading database')

	print('collecting assignments')
	assignment_tuples = collect_assignments(key_phrase_dict, score_list, trip, k=10)
	print('finished collecting')

	with open('assignment_tuples.txt', 'w') as file_tuples:
		for line in assignment_tuples:
			file_tuples.write('{}\n'.format(line))