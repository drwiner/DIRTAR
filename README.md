David Winer IE task: DIRTAR: "Discovery of Inference Rules from Text for Action Recognition"

Summary: DIRT algorithm (Lin and Pantel, 2001) with modifications (lemmas, constituency parse, slot-sim, slot-types, hypernyms, semantic-discrimination)


CODE FILES:

moviescript_crawler.py - collects movie corpus (from local path) and each document is inserted into a single document called movie_combo.txt.

sentence_parser.py - reads moviescripts from movie_comb.txt, splits into sentences, and each sentence into clauses (using constituency parse), output is "movie_cluases.txt"

semantic_parser.py - used for one of the experimental conditions - hand written frame-net style rules for discriminating candidate nouns from slots

dirtar.py - runs the dirt algorithm with all experimental conditions, and includes which X and Y slot dependencies are included for some of the experimental conditions. Running this file dumps the databases as pickle files, which are loaded for analysis by "assign_labels_moviedirt.py"

assign_labels_moviedirt.py - reads triple databases and reads (and parses with stanford parser) the test sentences (duel corpus sentences), in "IE_sent_key.txt". Outputs text files for each experimental condition where each line is a verb in the duel corpus, the guess, etc.

score_labels_dirtar.py - reads the experimental labels and calculates fscore, etc, and spits out files which are in "scored_labels" folder


FOLDERS:

extract_duel_sentences.zip - includes original duel corpus and data structures used to construct "IE_sent_key.txt" from excel file.

experimental_labels - folder containing the text files spit by assign_labels_moviedirt.py

scored_labels - folder containing the evaluation of each experimental condition, includes "total" and 1 per action of interest


Other Files:

IE_sent_key.txt - each line has a sentence from duel corpus, followed by "-#-", followed by list of action classes for that sentence.

movie_combo.txt - not attached (too large) contains combined text file for movies from Walker's cleaned imsdb database (https://nlds.soe.ucsc.edu/fc2)

movie_clauses.txt - movie_comb.txt separated into clause triples, where each slot in the triple has some extra annotations from the parse
