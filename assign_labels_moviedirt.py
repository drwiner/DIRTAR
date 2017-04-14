#!/usr/bin/python

import pickle
import dirtar as mdirt
from dirtar import Entry, rec_dd
from pycorenlp import StanfordCoreNLP
from functools import partial

from sentence_parser import Sentence, Word, parse_to_clauses
import sentence_splitter
import semantic_parser

def cleanLine(line):
	return ' '.join(line.split()) + ' '


def parse_scene_sents(file_name):
	# each entry in this list is a tuple of the form (clause, action_list)
	sentence_verb_actions = []
	with open(file_name) as sc_sents:

		for line in sc_sents:
			raw_sent, raw_actions = line.split('-#-')
			actions = raw_actions.split()
			# formats from raw sentence
			sent = sentence_splitter.split_into_sentences(text=raw_sent + '.')[0]

			print('Parsing sent: \'{}\' \n'.format(sent))
			s = digest(sent)
			if s is None:
				print('\tdiscontinued\n')
				continue

			# paths are tuples of the from (left_thing, verb_lemma, right_thing)
			verb_lemmas = [clause[0] for clause in Sentence(s).clauses]
			action_lemmas = [act for act in actions if act in ACTION_TYPES]

			# list of sentences of the from (verb_lemmas, action_lemmas)
			sentence_verb_actions.append((verb_lemmas, action_lemmas))

	return sentence_verb_actions


def assign_labels(db, mst_db, sents, K, output):

	for j, (verb_list, action_list) in enumerate(sents):
		for verb in verb_list:
			ranked_list = mdirt.most_similar_to(verb, db)
			best_action = None
			best_score = 0

			for k in K:
				top_k = ranked_list[:k]
				for action, action_ranked_list in mst_db.items():
					common = action_ranked_list[:k] & top_k
					if len(common) == 0:
						continue
					elif len(common) > 1:
						# choose the best score
						for common_lemma in common:
							score = mdirt.pathsimdb(common_lemma, action, db)
							if score > best_score:
								best_score = score
								best_action = common_lemma
					else:
						best_action = common.pop()
						best_score = (best_action, action)

				with open(k + '_' + output, 'a') as ona:
					ona.write('{}\t{}\t{}\t{}\t{}'.format(j, verb, best_action, best_score, action_list))

				best_action = None
				best_score = 0

mdirt_mst = mdirt.most_similar_with_multislot_with_semantic
def assign_labels_multi(db, mst_db, sents, K, output_names):
	# action_sim_dict[database] = mst_db
	#db, action_sim_dict[database]


	for j, verb_list, action_list in enumerate(sents):
		for verb in verb_list:
			ranked_tuple = mdirt_mst(verb, db)

			best_action = [None, None, None, None]
			comp_methods = [mdirt.pathSim_multiSlot, mdirt.pathSim_multiSlot, mdirt.weighted_pathSim_multiSlot, mdirt.weighted_pathSim_multiSlot]
			best_score = [0, 0, 0, 0]

			for i, ranked_list in enumerate(ranked_tuple):
				for k in K:
					top_k = ranked_list[:k]
					for action, action_ranked_tuple in mst_db.items():
						common = action_ranked_tuple[i][:k] & top_k
						if len(common) == 0:
							continue
						elif len(common) > 1:
							# choose the best score
							for common_lemma in common:
								score = comp_methods[i](common_lemma, action, db)
								if score > best_score[i]:
									best_score[i] = score
									best_action[i] = common_lemma
						else:
							best_action[i] = common.pop()
							best_score[i] = comp_methods[i](best_action[i], action)

					# append label to each file
					with open(k + '_' + output_names[i], 'a') as ona:
						ona.write('{}\t{}\t{}\t{}\t{}'.format(j, verb, best_action[i], best_score[i], action_list))

					best_action[i] = None
					best_score[i] = 0

def digest(rawd):
	return nlp(text=rawd)

def nlp_partial_sent(host_url):
	nlp_server = StanfordCoreNLP('http://localhost:9000')
	return partial(nlp_server.annotate, properties={'outputFormat': 'json'})


def nlp_partial(server_annotate, text):
	parse = server_annotate(text)
	try:
		return parse['sentences'][0]
	except:
		return None


if __name__ == '__main__':

	### Setup Stanford server parse function "nlp"
	annotater = nlp_partial_sent('http://localhost:9000')
	nlp = partial(nlp_partial, server_annotate=annotater)

	# import key paths (action-paths)
	ACTION_TYPES = 'shoot aim hit look stare walk fall draw cock'.split()


	# import test sentences
	print('parsing sentence key')
	# each item in the list is a tuple (phrase_list, action_list)
	verb_action_lemmas = parse_scene_sents('IE_sent_key.txt')
	with open('duel_corpus_verb_test_list.txt', 'w') as dv:
		for i, (verb_list, action_list) in enumerate(verb_action_lemmas):
			dv.write('{}\t{}\t{}'.format(i, verb_list, action_list))
	print('finished parsing test sentence')


	# Load databases from storage for use
	s_names = ['tstream', 'ctstream', 'ftstream', 'fctstream', 'wstream', 'mstream']
	output_names = ['SVO', 'SVO_corrected', 'SVO_filtered', 'SVO_filtered_corrected', 'SVO_hypernyms', 'NONE']
	output_name_dict = dict(zip(s_names, output_names))
	prefix = 'dirtar_database_'
	suffix = '.pkl'

	action_sim_dict = dict()
	print('Finding most-similar-to action lemmans')
	for database in s_names:
		print(database)
		with open(prefix + database + suffix, 'rb') as tripdatabase:
			db = pickle.load(tripdatabase)
		action_sim_dict[database] = dict()

		for action in ACTION_TYPES:
			if database != 'mstream':
				action_sim_dict[database][action] = mdirt.most_similar_to(action, db)
			else:
				g_regular, g_semantic, w_regular, w_semantic = mdirt.most_similar_with_multislot_with_semantic(action, db)
				action_sim_dict[database][action] = (g_regular, g_semantic, w_regular, w_semantic)


	print('assigning labels')
	K = [10, 35, 100]
	for database in s_names:
		print(database)
		with open(prefix + database + suffix, 'rb') as tripdatabase:
			db = pickle.load(tripdatabase)
		if database == 'mstream':
			file_name_outputs = ['multi_reg_geo.txt', 'multi_sem_geo.txt', 'multi_reg_w.txt', 'multi_sem_w.txt']
			assign_labels_multi(db, action_sim_dict[database], verb_action_lemmas, K, output=file_name_outputs)
		else:
			output_file_name = output_name_dict[database]
			assign_labels(db, action_sim_dict[database], verb_action_lemmas, K, output=output_file_name + '.txt')