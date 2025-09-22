# Written by David Winer
# 2017 - 04 - 15

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

# action_remap_dict = {'fire':'fire', 'aim': 'aim', 'hit': 'get-shot', 'look': 'look-at', 'stare': 'stare-at',
	                     # 'walk': 'walk', 'fall': 'fall', 'draw': 'draw', 'cock': 'cock'}
action_remap_dict = {'fire':'fire', 'aim': 'aim', 'get-shot': 'hit', 'look-at': 'look', 'stare-at': 'stare',
	                     'walk': 'walk', 'walk-to' : 'walk', 'walk-from': 'walk',
	                 'face' : 'look', 'turn' : 'look', 'face-to': 'look', 'turn-to':'look', 'look-from-to' : 'look',
	                 'turn-from-to' : 'look', 'face-from-to': 'look',
	                 'walk-to-from': 'walk', 'arrive' : 'walk', 'leave': 'walk',
	                 'fall': 'fall', 'draw': 'draw', 'cock': 'cock'}


folder = 'vc_labels//'

# ATYPES = set(action_remap_dict.values())
# action_remap_dict = {'fire':'fire', 'aim': 'aim', 'hit': 'get-shot', 'look': 'look-at', 'stare': 'stare-at',
	                     # 'walk': 'walk', 'fall': 'fall', 'draw': 'draw', 'cock': 'cock', None: 'None'}


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
			verb_lemmas = [clause[1].strip() for clause in Sentence(s).clauses]
			action_lemmas = [action_remap_dict[act] for act in actions if act in action_remap_dict.keys()]

			# list of sentences of the from (verb_lemmas, action_lemmas)
			sentence_verb_actions.append((verb_lemmas, action_lemmas))

	return sentence_verb_actions


def assign_labels_mst_psim(db, mst_db, sents, K, output, k_most_sim, psim):

	for j, (verb_list, action_list) in enumerate(sents):
		for verb in verb_list:
			print(verb)

			ranked_list = k_most_sim(verb, db)

			# ain't gonna work kid
			if ranked_list is None:
				for k in K:
					with open(folder + str(k) + '_' + str(output), 'a') as ona:
						ona.write('{}\t{}\t{}\t{}\t{}\n'.format(j, verb, None, 0, 0, action_list))
				continue

			for k in K:
				best_action = None
				best_score = 0
				exact_match = 0
				tk_list = [item[0] for item in ranked_list[:k]]
				top_k = set(tk_list)
				if verb in mst_db.keys():
					exact_match = 1
					best_action = verb
					best_score = 1.0
					with open(folder + str(k) + '_' + str(output), 'a') as ona:
						ona.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(j, verb, best_action, exact_match, best_score, action_list))
					continue

				found = 0
				for tk in tk_list:
					if tk in ACTION_TYPES:
						best_action = tk
						best_score = 1.0
						found = 1
						break

				if not found:
					for action, action_ranked_list in mst_db.items():
						art = {item[0] for item in action_ranked_list[:k]}
						common = art & top_k
						if len(common) == 0:
							continue
						elif len(common) > 1:
							# choose the best score
							for common_lemma in common:
								score = psim(common_lemma, action, db)
								if score > best_score:
									best_score = score
									best_action = action
						else:
							common_lemma = common.pop()
							score = psim(common_lemma, action, db)
							if score > best_score:
								best_score = score
								best_action = action

				print('best action: {}'.format(best_action))

				with open(folder + str(k) + '_' + str(output), 'a') as ona:
					ona.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(j, verb, best_action, exact_match, best_score, action_list))


def assign_labels(db, mst_db, sents, K, output):

	for j, (verb_list, action_list) in enumerate(sents):
		for verb in verb_list:
			print(verb)

			ranked_list = mdirt.most_similar_to(verb, db)

			# ain't gonna work kid
			if ranked_list is None:
				for k in K:
					with open(folder + str(k) + '_' + str(output), 'a') as ona:
						ona.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(j, verb, None, 0, 0, action_list))
				continue

			for k in K:
				best_action = None
				best_score = 0
				exact_match = 0
				tk_list = [item[0] for item in ranked_list[:k]]
				top_k = set(tk_list)
				if verb in mst_db.keys():
					exact_match = 1
					best_action = verb
					best_score = 1.0
					with open(folder + str(k) + '_' + str(output), 'a') as ona:
						ona.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(j, verb, best_action, exact_match, best_score, action_list))
					continue

				found = 0
				for tk in tk_list:
					if tk in ACTION_TYPES:
						best_action = tk
						best_score = 1.0
						found = 1
						exact_match = 1
						break

				if not found:
					for action, action_ranked_list in mst_db.items():
						art = {item[0] for item in action_ranked_list[:k]}
						common = art & top_k
						if len(common) == 0:
							continue
						elif len(common) > 1:
							# choose the best score
							for common_lemma in common:
								score = mdirt.pathSimdb(common_lemma, action, db)
								if score > best_score:
									best_score = score
									best_action = action
						else:
							common_lemma = common.pop()
							score = mdirt.pathSimdb(common_lemma, action, db)
							if score > best_score:
								best_score = score
								best_action = action

				print('best action: {}'.format(best_action))

				with open(folder + str(k) + '_' + str(output), 'a') as ona:
					ona.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(j, verb, best_action, exact_match, best_score, action_list))


mdirt_mst = mdirt.most_similar_with_multislot_with_semantic
def assign_labels_multi(db, mst_db, sents, K, output_names):
	# mst_db = action_sim_dict[database]

	for j, (verb_list, action_list) in enumerate(sents):
		for verb in verb_list:
			print(verb)
			two_lists = mdirt.most_similar_with_multislot_with_semantic(verb, db, semantic=0)

			# ain't gonna work kid
			if two_lists is None:
				for k in K:
					for output in output_names:
						with open(folder + str(k) + '_' + str(output), 'a') as ona:
							ona.write('{}\t{}\t{}\t{}\t{}\n'.format(j, verb, None, 0, action_list))
				continue

			best_action = [None, None, None, None]
			best_score = [0, 0, 0, 0]
			cmp_lists = [two_lists[0], two_lists[1], two_lists[0], two_lists[1]]
			path_methods = [mdirt.pathSim_multiSlot, mdirt.weighted_pathSim_multiSlot,
			                mdirt.pathSim_multiSlot, mdirt.weighted_pathSim_multiSlot]

			for k in K:
				for i in range(4):
					best_action[i] = None
					best_score[i] = 0
					exact_match = 0
					# item is tuple (verb, score)
					tk_list = [item[0] for item in cmp_lists[i][:k]]
					top_k = set(tk_list)
					if verb in mst_db.keys():
						exact_match = 1
						best_action[i] = verb
						best_score[i] = 1.0
						with open(folder+str(k) + '_' + output_names[i], 'a') as ona:
							ona.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(j, verb, best_action[i], str(exact_match),
							                                            str(best_score[i]), action_list))
						continue

					found = 0
					for tk in tk_list:
						if tk in ACTION_TYPES:
							best_action = tk
							best_score = 1.0
							found = 1
							exact_match = 1
							break

					if not found:
						for action, action_ranked_tuple in mst_db.items():
							art = {item[0] for item in action_ranked_tuple[i][:k]}
							common = art & top_k
							if len(common) == 0:
								continue
							elif len(common) > 1:
								# choose the best score
								for common_lemma in common:
									score = path_methods[i](common_lemma, action, db)
									if score > best_score[i]:
										best_score[i] = score
										best_action[i] = action
							else:
								common_lemma = common.pop()
								score = path_methods[i](common_lemma, action, db)
								if score > best_score[i]:
									best_score[i] = score
									best_action[i] = action

					# append label to each file
					print('best action: {}'.format(best_action[i]))

					with open(folder+str(k) + '_' + output_names[i], 'a') as ona:
						ona.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(j, verb, best_action[i], str(exact_match), str(best_score[i]), action_list))


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
	ACTION_TYPES = 'fire aim hit look stare walk fall draw cock'.split()


	# import test sentences
	print('parsing sentence key')
	# each item in the list is a tuple (phrase_list, action_list)
	verb_action_lemmas = parse_scene_sents('IE_sent_key.txt')


	# Load databases from storage for use
	s_names = ['tstream', 'ctstream', 'ftstream', 'fctstream', 'wstream', 'cmstream', 'mstream']
	output_names = ['SVO', 'SVO_corrected', 'SVO_filtered', 'SVO_filtered_corrected', 'SVO_hypernyms', 'multi_corrected_w', 'NONE']
	# s_names = ['mstream']
	# output_name = ['NONE']
	output_name_dict = dict(zip(s_names, output_names))
	prefix = 'dirtar_database_'
	suffix = '.pkl'


	print('Finding most-similar-to action lemmas')
	# K = [10, 15, 35]
	K = [15,20,25]

	for database in s_names:

		print('loading db: {}'.format(database))
		with open(prefix + database + suffix, 'rb') as tripdatabase:
			db = pickle.load(tripdatabase)

		action_sim_dict = dict()

		print('Finding most-similar-to action lemmas {}'.format(database))
		for action in ACTION_TYPES:
			if database not in  {'mstream', 'cmstream'}:
				action_sim_dict[action] = mdirt.most_similar_to(action, db)
			elif database == 'mstream':
				g_regular, g_semantic, w_regular, w_semantic = mdirt.most_similar_with_multislot_with_semantic(action, db)
				action_sim_dict[action] = (g_regular, g_semantic, w_regular, w_semantic)
			else:

				action_sim_dict[action] = mdirt.most_similar_wpathsim(action, db)

		print('assigning labels: {}'.format(database))
		if database == 'mstream':
			file_name_outputs = ['multi_reg_w.txt']
			assign_labels_multi(db, action_sim_dict, verb_action_lemmas, K, output_names=file_name_outputs)
		elif database == 'cmstream':
			assign_labels_mst_psim(db, action_sim_dict, verb_action_lemmas, K, 'multi_corrected_w.txt', mdirt.most_similar_wpathsim, mdirt.weighted_pathSim_multiSlot)
		else:
			output_file_name = output_name_dict[database]
			assign_labels(db, action_sim_dict, verb_action_lemmas, K, output=output_file_name + '.txt')


		# import score_labels_dirtar
