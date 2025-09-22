#!/usr/bin/python

def clean_action_list(action_string_list):
	action_list = []
	for item in action_string_list:
		action = clean_action_item(item)
		action_list.append(action)
	return action_list

def clean_action_item(action_string):
	return action_string.strip().split('\'')

def read_tuples(fname):
	with open(fname) as fn:
		for line in fn:
			line_sp = line.split(',')
			if line_sp[0].strip().split('\'')[1] == 'unlabeled':
				act = line_sp[2].split('\'')[1]
				penal = float(line_sp[3].strip().split('\'')[1])
				if penal > 1:
					num_missed_dict[act] += 1
				else:
					num_missed_dict[act] += penal
			else:
				labeled_as = clean_action_item(line_sp[2])[1]
				correct = int(line_sp[-1].strip()[1])
				if correct:
					num_correct_dict[labeled_as] += 1
				else:
					num_incorrect_dict[labeled_as] += 1


def score(actions):
	for action in actions:
		correct = num_correct_dict[action]
		try:
			recall_dict[action] = correct / (correct + num_missed_dict[action])
		except:
			recall_dict[action] = 0
		try:
			precision_dict[action] = correct / (correct + num_incorrect_dict[action])
		except:
			precision_dict[action] = 0


if __name__ == '__main__':
	import collections
	file_name = 'assignment_tuples.txt'

	num_missed_dict = collections.defaultdict(int)
	num_correct_dict = collections.defaultdict(int)
	num_incorrect_dict = collections.defaultdict(int)

	print('reading tuples')
	read_tuples(file_name)
	print('finished reading tuples')

	actions = num_missed_dict.keys() | num_correct_dict.keys() | num_incorrect_dict.keys()

	recall_dict = collections.defaultdict(int)
	precision_dict = collections.defaultdict(int)

	print('scoring actions')
	score(actions)
	print('finished scoring actions')

	with open('moviedirt_results.txt', 'w') as results:
		for action in actions:
			precision = precision_dict[action]
			recall = recall_dict[action]
			if precision == 0 or recall == 0:
				fscore = 0
			else:
				fscore = (2 * precision * recall) / (precision + recall)
			results.write('{}\n'.format(action))
			results.write('Num Labeled: {}\nNum Unlabeled {}\n'.format(num_correct_dict[action] + num_incorrect_dict[action], num_missed_dict[action]))
			results.write('Precision: {} \n'.format(precision))
			results.write('Recall: {} \n'.format(recall))
			results.write('Fscore: {} \n'.format(fscore))
			results.write('\n')
