David Winer IE task: DIRTAR: "Discovery of Inference Rules from Text for Action Recognition"

Summary: Classic DIRT algorithm (Lin and Pantel, 2001) but slots are not "X" and "Y", instead are dependency slots like "Subj", "DObj", and "PP-by". Special "semantic discrimination" module filters candidates for test.

The movie corpus is collected with the moviescript_crawler.py file (from local path) and each document is inserted into a single document called movie_combo.txt. The text was when formatted with "format_corpus.txt" with output movie_corpus.txt, whose formatting is similar to that as the input file in the DIRT assignment. The DIRTAR algorithm is run with moviedirt.py and a triple database is dumped with pickle as "trip_database.pkl"

The "movie_output10.txt" is the output to the moviedirt.py execution where the min-freq is 5 and the number of most-similar phrases to output is 10. The test phrases are in the file "key_phrases" and are the handwritten action phrases.

The duel corpus is parsed and saved as "IE_sent_key.txt" where each line has the sentence, followed by "-#-", followed by the list of action classes for that sentence. The triple database is used to assign labels to clauses in each sentence in the document "assign_labels_moviedirt.py". In this document is the "find-best-action" algorithm; the method is called "collect_assignments". The assignments are saved in the document "assignment_tuples.txt". The "score_labels_moviedirt.py" script reads these assignments and scores them to calculate precision, recall, and fscore.