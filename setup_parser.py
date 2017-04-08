# the_model_path = 'D://Documents//Python//NLP//stanford-corenlp-full-2016-10-31//stanford-corenlp-full-2016-10-31//stanford-corenlp-3.7.0-models//edu//stanford//nlp//models//lexparserenglishPCFG.ser.gz'
# jars = 'D://Documents//Python//NLP//stanford-parser-full-2016-10-31//stanford-parser-full-2016-10-31//jars//'
# import os
# from nltk.parse import stanford
# os.environ['STANFORD_PARSER'] = jars
# os.environ['STANFORD_MODELS'] = jars
# parser = stanford.StanfordParser(model_path=the_model_path)
# sentences = parser.raw_parse_sents(("Hello, My name is Melroy.", "What is your name?"))
# print(sentences)
#
# for line in sentences:
# 	for sentence in line:
# 		sentence.draw()

from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
text = 'You and I are mortal but rock and roll will never die.'
output = nlp.annotate(text, properties={'outputFormat':'json'})
print('you there')

output2['sentences'][0]['tokens'][4]

"""
output['sentences'][x] --> sentences
sentence['tokens'] is a list whose elements are tokens
	each token has pos, word, ner, lemma, "speaker"?


"""
#outpu