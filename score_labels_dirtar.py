# Written by David Winer
# 2017 - 04 - 15

def clean_action_list(action_string_list):
	action_list = []
	for item in action_string_list:
		action = clean_action_item(item)
		action_list.append(action)
	return action_list

def clean_action_item(action_string):
	return action_string.strip().split('\'')

ACTION_TYPES = 'fire aim hit look stare walk fall draw cock'.split()

from collections import namedtuple
from collections import defaultdict

Label = namedtuple('Label', ['shot_num', 'verb', 'guess', 'score', 'action_set'])
import copy
class Scorecard:
	def __init__(self, value_type):
		self.correct = copy.deepcopy(value_type)
		self.incorrect = copy.deepcopy(value_type)
		self.misses = copy.deepcopy(value_type)
		self.exacts = copy.deepcopy(value_type)
		self.wrong_exacts = copy.deepcopy(value_type)


# folder = 'experimental_labels//'
folder = 'redo_labels_420//'
def parse(file_name):
	lines = []
	with open(folder+file_name, 'r') as fn:
		for j, line in enumerate(fn):
			line_sp = line.split('\t')
			action_list = answer_key[j]
			label = Label(int(line_sp[0]), line_sp[1].strip(), line_sp[2].strip(), float(line_sp[3]), action_list)
			lines.append(label)
	return lines

from random import randint
def random_baseline(file_name):
	cndts = list(ACTION_TYPES)
	lines = []
	with open('experimental_labels//' + file_name, 'r') as fn:
		for line in fn:
			line_sp = line.split('\t')
			if line_sp[-1].strip() == '[]':
				action_list = set()
			else:
				try:
					action_list = set(eval(line_sp[4].strip()))
				except:
					action_list = set(eval(line_sp[5].strip()))
			label = Label(int(line_sp[0]), line_sp[1].strip(), cndts[randint(0,len(cndts)-1)].strip(), 0.0, action_list)
			lines.append(label)

	return lines


def score_sent(labels, asc, tsc):
	action_checkmark_set = set(labels[0].action_set)
	for label in labels:
		# correct AND exact
		if label.verb in ACTION_TYPES and label.guess in label.action_set:
			tsc.correct += 1
			asc.correct[label.verb] += 1
			tsc.exacts += 1
			asc.exacts[label.verb] += 1
			if label.verb in action_checkmark_set:
				action_checkmark_set -={label.verb}
			continue
		if label.verb in ACTION_TYPES:
			tsc.wrong_exacts += 1
			asc.wrong_exacts[label.verb] += 1
		if label.guess in label.action_set:
			tsc.correct += 1
			asc.correct[label.guess] += 1
			if label.guess in action_checkmark_set:
				action_checkmark_set -= {label.guess}
			continue
		if label.guess == 'None' or label.guess.strip() == 'None':
			# print('None')
			continue
		tsc.incorrect += 1
		asc.incorrect[label.guess] += 1
	for action in action_checkmark_set:
		asc.misses[action] += 1
		tsc.misses += 1


def calculate(correct, incorrect, misses, exacts):
	labeled = correct + incorrect
	if labeled > 0:
		precision = correct / labeled
		exact_precision = exacts / labeled
	else:
		precision = 0
		exact_precision = 0
	if labeled > 0 or misses > 0:
		recall = correct / (labeled + misses)
	else:
		recall = 0

	if precision == 0 or recall == 0:
		fscore = 0
	else:
		fscore = 2 * (precision * recall) / (precision + recall)
	return labeled, precision, recall, exact_precision, fscore

if __name__ == '__main__':
	# K = [10,15, 35]*5 + [10,15, 35]*4
	# K = []*5
	K = [15]*5 + [20]*5 + [25]*5
	# types = ['SVO']*15 + ['multi']*12
	types = ['SVO']*15 #+ ['multi']*5
	# svo_suffices = ['', '_filtered', '_corrected', '_filtered_corrected', '_hypernyms']*3


	# K = [15] * 10 + [20] * 10 + [25] * 10
	# types = (['SVO'] * 5 + ['multi'] * 5) * 3

	svo_suffices = ['', '_filtered', '_corrected', '_filtered_corrected', '_hypernyms'] * 3
	# multi_suffices = ['_reg_geo', '_reg_w', '_sem_geo', '_sem_w']*3
	# multi_suffices = ['_corrected_w', '_reg_geo', '_reg_w', '_sem_geo', '_sem_w']
	multi_suffices = []
	suffices = svo_suffices + multi_suffices


	answer_key = dict()
	with open('experimental_labels//15_SVO.txt', 'r') as el:
		for i, line in enumerate(el):
			line_sp = line.split('\t')
			if line_sp[-1].strip() == '[]':
				action_list = set()
			else:
				try:
					action_list = set(eval(line_sp[5].strip()))
				except:
					action_list = set(eval(line_sp[4].strip()))
			answer_key[i] = action_list

	# file_names = ['{}_{}{}.txt'.format(K[i], types[i], suffices[i]) for i in range(27)]
	file_names = ['{}_{}{}.txt'.format(K[i], types[i], suffices[i]) for i in range(15)]
	scorekeeper = dict()
	for f_name in file_names:

		# action specific
		action_scorecard = Scorecard(defaultdict(int))

		# total
		total_scorecard = Scorecard(0)

		# parse lines to tuple of data stractures
		labels = parse(f_name)


		# separate into sentences
		sent_lines = []
		current_sent = 0
		for label in labels:
			if current_sent < label.shot_num:
				# process last sentence
				score_sent(sent_lines, action_scorecard, total_scorecard)
				# flush
				current_sent = label.shot_num
				sent_lines = [label]
			else:
				sent_lines.append(label)
		if len(sent_lines) > 0:
			score_sent(sent_lines, action_scorecard, total_scorecard)


		calc = calculate(total_scorecard.correct,
		                 total_scorecard.incorrect,
		                 total_scorecard.misses,
		                 total_scorecard.exacts)
		labeled, precision, recall, exact_precision, fscore = calc

		print(f_name)
		print('total labeled {}'.format(labeled))
		print('total precision: {}'.format(precision))
		print('total recall: {}'.format(recall))
		print('total exact precision: {}'.format(exact_precision))
		print('total fscore: {}'.format(fscore))
		print('\n')

		for action in ACTION_TYPES:
			a = action
			calc = calculate(action_scorecard.correct[a],
			                 action_scorecard.incorrect[a],
			                 action_scorecard.misses[a],
			                 action_scorecard.exacts[a])
			labeled, precision, recall, exact_precision, fscore = calc

			print(f_name)
			print('{} labeled {}'.format(action, labeled))
			print('{} precision: {}'.format(action, precision))
			print('{} recall: {}'.format(action, recall))
			print('{} exact precision: {}'.format(action, exact_precision))
			print('{} fscore: {}'.format(action, fscore))
			print('\n')

		scorekeeper[f_name] = (total_scorecard, action_scorecard)

	# grade random baseline as well, picking one action per sentence
	grading_example = file_names[0]
	labels = random_baseline(grading_example)
	action_scorecard = Scorecard(defaultdict(int))
	total_scorecard = Scorecard(0)

	# separate into sentences
	sent_lines = []
	current_sent = 0
	for label in labels:
		if current_sent < label.shot_num:
			# process last sentence
			score_sent(sent_lines, action_scorecard, total_scorecard)
			# flush
			current_sent = label.shot_num
			sent_lines = [label]
		else:
			sent_lines.append(label)
	if len(sent_lines) > 0:
		score_sent(sent_lines, action_scorecard, total_scorecard)

	calc = calculate(total_scorecard.correct,
	                 total_scorecard.incorrect,
	                 total_scorecard.misses,
	                 total_scorecard.exacts)
	labeled, precision, recall, exact_precision, fscore = calc

	print('Random:')
	print('total labeled {}'.format(labeled))
	print('total precision: {}'.format(precision))
	print('total recall: {}'.format(recall))
	print('total exact precision: {}'.format(exact_precision))
	print('total fscore: {}'.format(fscore))
	print('\n')

	for action in ACTION_TYPES:
		a = action
		calc = calculate(action_scorecard.correct[a],
		                 action_scorecard.incorrect[a],
		                 action_scorecard.misses[a],
		                 action_scorecard.exacts[a])
		labeled, precision, recall, exact_precision, fscore = calc

		print('random')
		print('{} labeled {}'.format(action, labeled))
		print('{} precision: {}'.format(action, precision))
		print('{} recall: {}'.format(action, recall))
		print('{} exact precision: {}'.format(action, exact_precision))
		print('{} fscore: {}'.format(action, fscore))
		print('\n')

	scorekeeper['random'] = (total_scorecard, action_scorecard)

	total_rows = []
	headers = ['name', 'labeled', 'exact_precision', 'precision', 'recall', 'fscore']
	total_rows.append(headers)
	for item in ['random'] + file_names:
		total_sc = scorekeeper[item][0]
		calc = calculate(total_sc.correct, total_sc.incorrect, total_sc.misses, total_sc.exacts)
		labeled, precision, recall, exact_precision, fscore = calc

		row = [item, labeled, exact_precision, precision, recall, fscore]
		total_rows.append(row)

	action_rows_dict = defaultdict(list)
	headers = ['name', 'action', 'labeled', 'exact_precision', 'precision', 'recall', 'fscore']
	for action in ACTION_TYPES:
		action_rows_dict[action].append(headers)
		for item in ['random'] + file_names:
			action_sc = scorekeeper[item][1]

			a = action
			calc = calculate(action_sc.correct[a], action_sc.incorrect[a], action_sc.misses[a], action_sc.exacts[a])
			labeled, precision, recall, exact_precision, fscore = calc

			row = [item, action, labeled, exact_precision, precision, recall, fscore]
			action_rows_dict[action].append(row)

	#
	#
	with open('redo_scored_labels_420//total_plot.txt', 'w') as tp:
		for row in total_rows:
			for item in row:
				tp.write('{}\t'.format(item))
			tp.write('\n')


	for action in ACTION_TYPES:
		with open('redo_scored_labels_420//{}_plot.txt'.format(action), 'w') as tp:
			for row in action_rows_dict[action]:
				for item in row:
					tp.write('{}\t'.format(item))
				tp.write('\n')

