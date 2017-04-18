import dirtar as mdirt
from dirtar import Entry, rec_dd
import pickle
from functools import partial


def just_get(action_lemmas, db, index_of):
	d = dict()
	for action in action_lemmas:
		# g_regular, g_semantic, w_regular, w_semantic
		score_tup = mdirt.most_similar_with_multislot_with_semantic(action, db)
		d[action] = score_tup[index_of]
	return d


def test_most_similar_to(z, db_name, db, action_lemma_doc, k_most_similar, method):

	test_d = method(action_lemma_doc, db)

	with open('movie_output-' + db_name + str(z), 'w') as ot:
		# ot.write(line1)
		# ot.write(line2)
		for lemma, paths in test_d.items():
			ot.write('\n')
			ot.write('MOST SIMILAR RULES FOR: "{}"\n'.format(lemma))
			if lemma not in db.keys():
				ot.write('This phrase is not in the triple database.\n')
			else:
				for j, i in enumerate(paths):
					if j >= k_most_similar:
						if paths[j] != paths[-1]:
							break
					p, score = paths[j]
					if score > 0:
						ot.write(str(j + 1) + '. \"%s\" %24.12f\n' % (p, score))
					else:
						break

if __name__ == '__main__':
	test_phrase_file = 'action_lemmas.txt'


	with open (test_phrase_file, 'r') as tpf:
		action_lemmas = [line.strip() for line in tpf]

	s_names = ['tstream', 'ctstream', 'ftstream', 'fctstream', 'wstream', 'mstream', 'mstream', 'mstream', 'mstream']
	# output_names = ['SVO', 'SVO_corrected', 'SVO_filtered', 'SVO_filtered_corrected', 'SVO_hypernyms'] + ['multi_reg_geo.txt', 'multi_sem_geo.txt', 'multi_reg_w.txt', 'multi_sem_w.txt']
	# output_name_dict = dict(zip(s_names, output_names))
	prefix = 'dirtar_database_'
	suffix = '.pkl'
	mst = mdirt.most_similar_to_test
	reg_multi_slot = mdirt.pathSim_multiSlot
	weighted_multi_slot = mdirt.weighted_pathSim_multiSlot
	# [partial(just_get, action_lemmas=action_lemmas, index_of=i) for i in range(4)]
	methods = [mst, mst, mst, mst, mst] + [partial(just_get, index_of=i) for i in range(4)]

	for i, database in enumerate(s_names):

		print('loading db: {}'.format(database))
		with open(prefix + database + suffix, 'rb') as tripdatabase:
			db = pickle.load(tripdatabase)


		test_most_similar_to(i, database, db, action_lemmas, 10, methods[i])



