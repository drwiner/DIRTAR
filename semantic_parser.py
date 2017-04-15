"""
Written by David Winer 2017 - 04 - 15
each action lemma is assigned a set of rules for left and right slots inspired by framenet
"""

from nltk.corpus import wordnet as wn
from collections import defaultdict
from dirtar import Entry

class ActionFrame:
	def __init__(self, lemma):
		self.lemma = lemma
		self.restrictions = {'X': defaultdict(set), 'Y': defaultdict(set)}

	def lemma(self):
		return self.lemma + '_frame'

	def add_rule(self, slot_pos, dep_type_set, sem_token_list):
		for dep_type in dep_type_set:
			for token in sem_token_list:
				self.restrictions[slot_pos][dep_type].add(token)

	def filter(self, slot, dep, noun, ner):
		if noun in REMAP.keys():
			wn_token = REMAP[noun]
		elif ner == 'PERSON':
			wn_token = person
		else:
			tokens = wn.synsets(noun, wn.NOUN)
			if len(tokens) == 0:
				if len(self.restrictions[slot][dep]) == 0:
					# then it doesn't matter
					return True
				return False

			wn_token = tokens[0]

		# returns True if valid and False otherwise
		sem_types = self.restrictions[slot][dep]

		# if there's no restrictions on this slot, then let it pass
		if len(sem_types) == 0:
			return True

		for sem_type in sem_types:
			dist = wn_token.shortest_path_distance(sem_type)
			if dist < 6:
				return True

		return False

gun = wn.synsets('gun', wn.NOUN)[0]
body_part = wn.synsets('body_part', wn.NOUN)[0]
location = wn.synsets('location')[0]
way = wn.synsets('path')[0]
bullet = wn.synsets('bullet')[0]
ammo = wn.synsets('ammo')[0]
foot = wn.synsets('foot', wn.NOUN)[0]
hand = wn.synsets('hand', wn.NOUN)[0]
person = wn.synsets('person', wn.NOUN)[0]
eyes = wn.synsets('eyes')[0]
face = wn.synsets('face')[0]
head = wn.synsets('head')[0]
gunman = wn.synsets('gunman')[0]
victim = wn.synsets('victim')[0]
tree = wn.synsets('tree')[0]
sky = wn.synsets('sky')[0]
ground = wn.synsets('ground')[0]
floor = wn.synsets('floor')[0]
wall = wn.synsets('wall')[0]
crowd = wn.synsets('crowd')[0]
bottle = wn.synsets('bottle')[0]


REMAP = {'he': person, 'she': person, 'they': person, 'it': person, 'we': person}

LEFT_DEPS = ['nsubj', 'nsubj:xsubj', 'nsubjpass', 'nmod:poss']
RIGHT_DEPS = ['dobj', 'iobj', 'nmod:at', 'nmod:from', 'nmod:by', 'nmod:to', 'nmod:agent', 'nmod:in', 'nmod:into', 'nmod:poss', 'nmod:through', 'nmod:on', 'nmod:across', 'nmod:over', 'nmod:away_from']


# inspired by framenet

shoot = ActionFrame('shoot')
shoot.add_rule('X', {'nsubj', 'nsubj:xsubj'}, [person, gunman])

shoot.add_rule('Y', {'nmod:from'}, [gun, ground, floor, location, crowd, tree])
shoot.add_rule('Y', {'nmod:with'}, [gun, ammo, bullet])


aim = ActionFrame('aim')
aim.add_rule('X', {'nsubj', 'nsubj:xsubj'}, [person, gunman])
# aim.add_rule('X', {'nsubjpass', 'nmod:poss'}, [person, victim, gunman, body_part, wall, head, ground, eyes, sky, floor, crowd, bottle])
# aim.add_rule('Y', {'dobj', 'nmod:at'},
#                [person, victim, gun, floor, ground, crowd, sky, foot, head, eyes, face, wall, bottle, tree,
#                 body_part])
aim.add_rule('Y', {'nmod:from'}, [ground, floor, location, crowd, tree])
aim.add_rule('Y', {'nmod:with'}, [gun])


hit = ActionFrame('hit')
hit.add_rule('X', {'nsubj', 'nsubj:xsubj'}, [person, gunman, bullet])
# hit.add_rule('Y', {'dobj'}, [person, body_part, foot, eyes, head, face, hand, bottle])


fall = ActionFrame('fall')
fall.add_rule('Y', {'nmod:to'}, [ground, floor, location])

look = ActionFrame('look')
look.add_rule('X', {'nsubj', 'nsubj:xsubj'}, [person, gunman, victim, eyes, face, head])

stare = ActionFrame('stare')
stare.add_rule('X', {'nsubj', 'nsubj:xsubj'}, [person, gunman, victim, eyes, face, head])

walk = ActionFrame('walk')
walk.add_rule('X', {'nsubj', 'nsubj:xsubj'}, [gunman, person, foot, victim])
walk.add_rule('X', {'nmod:from', 'nmod:away_from'}, [person, location, ground, way, floor, crowd, wall])
walk.add_rule('X', {'nmod:to'}, [person, location, ground, way, floor, crowd, wall])

draw = ActionFrame('draw')
draw.add_rule('X', {'nsubj', 'nsubj:xsubj'}, [person, gunman, hand])
draw.add_rule('Y', {'dobj', 'iobj'}, [gun])

cock = ActionFrame('cock')
cock.add_rule('X', {'nsubj', 'nmod:poss', 'nsubj:xsubj'}, [person, gunman, hand])
cock.add_rule('X', {'nsubjpass'}, [gun])
cock.add_rule('Y', {'dobj', 'iobj'}, [gun])

speak = ActionFrame('speak')
speak.add_rule('X', {'nsubj', 'nsubj:xsubj', 'nsubjpass'}, [person, gunman, victim])

action_frames = [shoot, aim, hit, fall, look, stare, walk, draw, cock, speak]
action_frame_dict = dict(zip([action.lemma for action in action_frames], action_frames))

def filter_action_lemma(lemma, db):
	# entry_db whose keys are slots and whose values are entries, described as follows:

	if lemma not in action_frame_dict.keys():
		print('path \'{}\' not found'.format(lemma))
		return
	frame = action_frame_dict[lemma]
	slots = db[lemma].keys()
	new_lemma = lemma + '_lemma'
	db[new_lemma] = dict()

	for slot in slots:
		slot_pos = slot[0].upper()
		db[new_lemma][slot] = dict()
		for noun, entry in db[lemma][slot].items():
			if frame.filter(slot_pos, entry.dep, entry.word, entry.ner):
				db[new_lemma][slot][noun] = entry

	return new_lemma


def get_action_lemma_entries():
	return action_frame_dict

if __name__ == '__main__':

	# semantic tokens
	pass

