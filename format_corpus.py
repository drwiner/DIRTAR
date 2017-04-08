import spacy

def read_corpus(file_name):
	print('reading')
	raw_doc = ''
	with open(file_name) as fn:
		# raw_doc += ' '.join(fn.readlines())
		for line in fn:
			sp = line.split()
			raw_doc += ' '.join(wrd.strip() for wrd in sp if wrd != '\n')
			raw_doc += ' '
	print('finished reading, now parsing nlp style')
	return nlp(raw_doc)

def noun_chunk_index_list(nchunks):
	indices = []
	index_chunk_dict = dict()
	for nchunk in nchunks:
		indices.extend(range(nchunk.start, nchunk.end))
		index_chunk_dict[nchunk.end-1] = nchunk #.root
	return indices, index_chunk_dict

symb_dict = dict()
symb_dict['\"'] = '>SQUOTE'
symb_dict['\''] = '>SQUOTE'
symb_dict['&'] = '>AMPERSAND'
symb_dict[','] = '>COMMA'
symb_dict['('] = '>LPAREN'
symb_dict[')'] = '>RPAREN'
symb_dict['-'] = '>MINUS'
symb_dict['--'] = '>MINUS'
symb_dict['.'] = '>PERIOD'


def parse_sent(sent):
	parse_list = []
	s_doc = nlp(sent.text)
	nchunks = list(s_doc.noun_chunks)
	nchunk_indices, nchunk_dict = noun_chunk_index_list(nchunks)
	for i, token in enumerate(sent):
		# if the index is a noun chunk, ignore, we will just insert all words between them? keep an origi
		if i in nchunk_indices:
			if i in nchunk_dict.keys():
				parse_list.append(nchunk_dict[i])
		else:
			if token.orth_ in symb_dict.keys():
				parse_list.append(symb_dict[token.orth_])
			else:
				parse_list.append(token)
	parse_list.append('<EOS')
	print(parse_list)
	return parse_list


def parse_to_output(sents, file_name):
	# open the file name
	with open(file_name, 'w') as fn:
		for snt in sents:
			for elm in parse_sent(snt):
				if type(elm) is spacy.tokens.span.Span:
					fn.write('{} : {}'.format('NP', elm.orth_))
				elif type(elm) is str:
					fn.write('WORD : {}'.format(elm))
				else:
					if elm.pos_ == 'SPACE':
						continue
					fn.write('{} : {}'.format(elm.pos_, elm.orth_))
				fn.write('\n')


if __name__ == '__main__':
	print('loading english to spaCy')
	nlp = spacy.load('en')

	file_name = 'movie_combo.txt'
	print('formatting corpus: {}'.format(file_name))
	doc = read_corpus(file_name)
	# with open(file_name) as fn:
	# 	print('reading')
	# 	doc = nlp(fn)
	print('done parsing, now extracting sentences')
	# snts = [[doc[i] for i in range(span.start, span.end)] for span in doc.sents]
	output_name = 'movie_corpus.txt'
	print('parsing to output file {}'.format(output_name))
	parse_to_output(doc.sents, output_name)
