"""
Modern implementation of the DIRTAR algorithm
"""

import logging
from collections import namedtuple, Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set, Union
from math import log2, sqrt
import operator
import pickle
import functools
from pathlib import Path

import nltk
from nltk.corpus import wordnet as wn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
LEFT_DEPS = ['nsubj', 'nsubj:xsubj', 'nsubjpass', 'nmod:poss']
REVERSIBLE_LEFTS = ['nsubjpass']
RIGHT_DEPS = ['iobj', 'dobj', 'nmod:at', 'nmod:from', 'nmod:by', 'nmod:to',
              'nmod:agent', 'nmod:in', 'nmod:into', 'nmod:poss', 'nmod:through',
              'nmod:on', 'nmod:across', 'nmod:over', 'nmod:away_from']
REVERSIBLE_RIGHTS = ['nmod:agent', 'nmod:by']
MULTI_SLOTS = LEFT_DEPS + RIGHT_DEPS

# Named tuples and data classes
Triple = namedtuple('Triple', ['X', 'path', 'Y'])

@dataclass
class DependencyInfo:
    """Information about dependency relationships"""
    noun: str
    dep: str
    ner: str

@dataclass
class Entry:
    """Database entry for DIRT algorithm"""
    path: Optional[str] = None
    slot: Optional[str] = None
    word: Optional[str] = None
    count: int = 1
    mi: Optional[float] = None
    dep: Optional[str] = None
    ner: Optional[str] = None

    def __post_init__(self):
        if self.word and self.slot and self.path:
            self._hash = hash((self.path, self.slot, self.word))
        else:
            self._hash = None

    def __hash__(self):
        return self._hash or 0

    def update_count(self):
        """Increment the count for this entry"""
        self.count += 1


class WordNetHelper:
    """Helper class for WordNet operations"""

    def __init__(self):
        try:
            self.person_synset = wn.synsets('person', wn.NOUN)[0]
        except IndexError:
            logger.warning("WordNet 'person' synset not found")
            self.person_synset = None

    def replace_word(self, word: str, ner: str) -> str:
        """Replace word with WordNet hypernym or person"""
        synsets = wn.synsets(word, wn.NOUN)

        if len(synsets) == 0 or ner == 'PERSON':
            return self.person_synset.lemma_names()[0] if self.person_synset else word

        h_paths = synsets[0].hypernym_paths()[0]
        if len(h_paths) < 6:
            return h_paths[-1].lemma_names()[0]
        return h_paths[5].lemma_names()[0]


class DIRTARProcessor:
    """Main DIRTAR algorithm processor"""

    def __init__(self, min_freq: int = 5):
        self.min_freq = min_freq
        self.wordnet_helper = WordNetHelper()
        self.streams = {
            'tstream': [],      # Basic triple stream
            'ctstream': [],     # Collapsed triple stream
            'ftstream': [],     # Filtered triple stream
            'fctstream': [],    # Filtered collapsed triple stream
            'wstream': [],      # WordNet-enhanced stream
            'cmstream': [],     # Corrected multi-slot stream
        }
        self.databases = {}

    def clean_line(self, line: str) -> str:
        """Clean and normalize a line of text"""
        return ' '.join(line.split()) + ' '

    def decide_swap(self, x: str, y: str, x_dep: str, y_dep: str) -> Tuple[str, str]:
        """Decide whether to swap X and Y based on dependency types"""
        if x_dep != 'none' and x_dep in REVERSIBLE_LEFTS:
            return y, x
        if y_dep != 'none' and y_dep in REVERSIBLE_RIGHTS:
            return y, x
        return x, y

    def read_corpus(self, clause_file: Union[str, Path]) -> None:
        """Read and process corpus file into various streams"""
        clause_file = Path(clause_file)

        if not clause_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {clause_file}")

        logger.info(f"Reading corpus from {clause_file}")

        with open(clause_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i % 10000 == 0:
                    logger.info(f"Processed {i} lines")

                try:
                    self._process_line(line.strip())
                except Exception as e:
                    logger.warning(f"Error processing line {i}: {e}")
                    continue

    def _process_line(self, line: str) -> None:
        """Process a single line from the corpus"""
        entries = line.split(',')

        # Handle malformed lines
        if len(entries) != 3:
            entries = self._fix_malformed_line(entries)
            if len(entries) != 3:
                return

        try:
            x_pieces = entries[0].split(' - ')
            y_pieces = entries[2].split(' - ')

            x = x_pieces[0].split('(')[1].strip().lower()
            y = y_pieces[0].split('(')[1].strip().lower()
            path = entries[1].strip()

            # Extract dependency and NER information
            x_dep = x_pieces[-1].split(')')[0].strip()
            x_ner = x_pieces[2].strip()
            y_dep = y_pieces[-1].split(')')[0].strip()
            y_ner = y_pieces[2].strip()

            # Create various stream entries
            self._add_to_streams(x, y, path, x_dep, x_ner, y_dep, y_ner)

        except (IndexError, ValueError) as e:
            logger.warning(f"Error parsing line: {line} - {e}")

    def _fix_malformed_line(self, entries: List[str]) -> List[str]:
        """Attempt to fix malformed lines with commas in unexpected places"""
        if len(entries) > 3 and entries[0] == '(':
            return [f'(comma {entries[1]}', entries[2], entries[3]]
        elif len(entries) == 2:
            return [entries[0], entries[1], '(number - NP - NUMBER - dobj)']
        return entries

    def _add_to_streams(self, x: str, y: str, path: str,
                       x_dep: str, x_ner: str, y_dep: str, y_ner: str) -> None:
        """Add processed triple to various streams"""
        # Basic triple stream
        self.streams['tstream'].append(Triple(x, path, y))

        # WordNet-enhanced stream
        wn_x = self.wordnet_helper.replace_word(x, x_ner)
        wn_y = self.wordnet_helper.replace_word(y, y_ner)
        self.streams['wstream'].append(Triple(wn_x, path, wn_y))

        # Multi-slot stream with dependency info
        x_info = DependencyInfo(x, x_dep, x_ner)
        y_info = DependencyInfo(y, y_dep, y_ner)

        # Filtered streams based on dependency types
        if x_dep in LEFT_DEPS and y_dep in RIGHT_DEPS:
            self.streams['ftstream'].append(Triple(x, path, y))

            # Collapsed with potential swapping
            x_prime, y_prime = self.decide_swap(x, y, x_dep, y_dep)
            self.streams['fctstream'].append(Triple(x_prime, path, y_prime))

        # Collapsed triple stream (with potential swapping)
        x_prime, y_prime = self.decide_swap(x, y, x_dep, y_dep)
        self.streams['ctstream'].append(Triple(x_prime, path, y_prime))

        # Corrected multi-slot stream
        if x_prime == x:
            self.streams['cmstream'].append(Triple(x_info, path, y_info))
        else:
            self.streams['cmstream'].append(Triple(y_info, path, x_info))

    def apply_frequency_filter(self, stream_name: str) -> Tuple[int, int, int, int, List[Triple]]:
        """Apply minimum frequency filter to a stream"""
        stream = self.streams[stream_name]

        # Count path frequencies
        path_counter = Counter([t.path for t in stream])

        # Filter based on minimum frequency
        distinct_unfiltered = set(path_counter.keys())
        distinct_filtered = set(p for p in distinct_unfiltered
                              if path_counter[p] >= self.min_freq)
        filtered_instances = [t for t in stream
                            if path_counter[t.path] >= self.min_freq]

        logger.info(f"{stream_name}: {len(distinct_unfiltered)} -> {len(distinct_filtered)} "
                   f"distinct paths, {len(stream)} -> {len(filtered_instances)} instances")

        return (len(distinct_unfiltered), len(distinct_filtered),
                len(stream), len(filtered_instances), filtered_instances)

    def load_database(self, db: Dict, filtered_instances: List[Triple],
                     multi_slot: bool = False) -> None:
        """Load filtered instances into database structure"""
        for triple in filtered_instances:
            x, path, y = triple

            if path is None or path == 'none':
                continue

            if path not in db:
                db[path] = {}

            if multi_slot:
                self._load_multislot_triple(db[path], x, y, path)
            else:
                self._load_basic_triple(db[path], x, y, path)

    def _load_basic_triple(self, path_db: Dict, x: str, y: str, path: str) -> None:
        """Load basic triple into database"""
        # Process X slot
        if x is not None and str(x) != 'none':
            if 'X' not in path_db:
                path_db['X'] = {}

            if x in path_db['X']:
                path_db['X'][x].update_count()
            else:
                path_db['X'][x] = Entry(path, 'X', x)

        # Process Y slot
        if y is not None and str(y) != 'none':
            if 'Y' not in path_db:
                path_db['Y'] = {}

            if y in path_db['Y']:
                path_db['Y'][y].update_count()
            else:
                path_db['Y'][y] = Entry(path, 'Y', y)

    def _load_multislot_triple(self, path_db: Dict, x: DependencyInfo,
                              y: DependencyInfo, path: str) -> None:
        """Load multi-slot triple with dependency information"""
        # Process X with dependency-based slot name
        if x.noun is not None and x.noun != 'none':
            x_slot_name = f'x_{x.dep}'
            if x_slot_name not in path_db:
                path_db[x_slot_name] = {}

            if x.noun in path_db[x_slot_name]:
                path_db[x_slot_name][x.noun].update_count()
            else:
                path_db[x_slot_name][x.noun] = Entry(
                    path, x_slot_name, x.noun, dep=x.dep, ner=x.ner
                )

        # Process Y with dependency-based slot name
        if y.noun is not None and y.noun != 'none':
            y_slot_name = f'y_{y.dep}'
            if y_slot_name not in path_db:
                path_db[y_slot_name] = {}

            if y.noun in path_db[y_slot_name]:
                path_db[y_slot_name][y.noun].update_count()
            else:
                path_db[y_slot_name][y.noun] = Entry(
                    path, y_slot_name, y.noun, dep=y.dep, ner=y.ner
                )

    def calculate_mutual_information(self, db: Dict, word_slot_count: Dict,
                                   slot_count: Dict, multi_slot: bool = False) -> None:
        """Calculate mutual information for all entries"""
        if multi_slot:
            self._calculate_mi_multislot(db, word_slot_count, slot_count)
        else:
            self._calculate_mi_basic(db, word_slot_count, slot_count)

    def _calculate_mi_basic(self, db: Dict, word_slot_count: Dict, slot_count: Dict) -> None:
        """Calculate MI for basic two-slot database"""
        # First pass: count occurrences
        for path, path_data in db.items():
            for slot in ['X', 'Y']:
                if slot in path_data:
                    for word, entry in path_data[slot].items():
                        count = entry.count
                        slot_count[slot] = slot_count.get(slot, 0) + count

                        if word not in word_slot_count:
                            word_slot_count[word] = {'X': 0, 'Y': 0}
                        word_slot_count[word][slot] += count

        # Second pass: calculate MI
        for path, path_data in db.items():
            for slot in ['X', 'Y']:
                if slot in path_data:
                    for entry in path_data[slot].values():
                        entry.mi = self._mutual_information(
                            db, word_slot_count, slot_count,
                            entry.path, entry.slot, entry.word
                        )

    def _calculate_mi_multislot(self, db: Dict, word_slot_count: Dict,
                               slot_count: defaultdict) -> None:
        """Calculate MI for multi-slot database"""
        # First pass: count occurrences
        for path, path_data in db.items():
            for slot, slot_data in path_data.items():
                for word, entry in slot_data.items():
                    count = entry.count
                    slot_count[slot] += count

                    if word not in word_slot_count:
                        word_slot_count[word] = defaultdict(int)
                    word_slot_count[word][slot] += count

        # Second pass: calculate MI
        for path, path_data in db.items():
            for slot, slot_data in path_data.items():
                for entry in slot_data.values():
                    if entry.word == 'none':
                        entry.mi = 0
                        continue

                    entry.mi = self._mutual_information(
                        db, word_slot_count, slot_count,
                        entry.path, entry.slot, entry.word
                    )

    def _mutual_information(self, db: Dict, word_slot_count: Dict, slot_count: Dict,
                          path: str, slot: str, word: str) -> float:
        """Calculate mutual information for a specific entry"""
        try:
            # |p,s,w| - count of (path, slot, word)
            psw = db[path][slot][word].count

            # |p,s,*| - count of (path, slot, any_word)
            ps_ = sum(entry.count for entry in db[path][slot].values())

            # |*,s,w| - count of (any_path, slot, word)
            _sw = word_slot_count.get(word, {}).get(slot, 0)

            # |*,s,*| - count of (any_path, slot, any_word)
            _s_ = slot_count.get(slot, 0)

            # Avoid division by zero
            if psw == 0 or ps_ == 0 or _sw == 0 or _s_ == 0:
                return 0.0

            # Calculate MI
            mi = log2((psw * _s_) / (ps_ * _sw))
            return max(0.0, mi)  # Return 0 if negative

        except (KeyError, ZeroDivisionError, ValueError):
            return 0.0

    def save_database(self, db: Dict, name: str, output_dir: Path = None) -> None:
        """Save database to pickle file"""
        if output_dir is None:
            output_dir = Path('.')

        output_path = output_dir / f'dirtar_database_{name}.pkl'

        with open(output_path, 'wb') as f:
            pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Saved database to {output_path}")

    def process_corpus(self, corpus_file: Union[str, Path],
                      output_dir: Path = None) -> Dict[str, Dict]:
        """Main processing pipeline"""
        logger.info("Starting DIRTAR processing pipeline")

        # Read corpus
        self.read_corpus(corpus_file)

        # Process each stream
        results = {}

        # Process basic streams
        for stream_name in ['tstream', 'ctstream', 'ftstream', 'fctstream', 'wstream']:
            logger.info(f"Processing {stream_name}")

            # Apply frequency filter
            stats = self.apply_frequency_filter(stream_name)
            filtered_instances = stats[4]

            # Create database
            db = {}
            self.load_database(db, filtered_instances, multi_slot=False)

            # Calculate MI
            word_slot_count = {}
            slot_count = {}
            self.calculate_mutual_information(db, word_slot_count, slot_count, multi_slot=False)

            # Save database
            self.save_database(db, stream_name, output_dir)

            results[stream_name] = {
                'database': db,
                'stats': stats,
                'word_slot_count': word_slot_count,
                'slot_count': slot_count
            }

        # Process multi-slot stream
        logger.info("Processing cmstream (multi-slot)")
        stats = self.apply_frequency_filter('cmstream')
        filtered_instances = stats[4]

        db = {}
        self.load_database(db, filtered_instances, multi_slot=True)

        word_slot_count = {}
        slot_count = defaultdict(int)
        self.calculate_mutual_information(db, word_slot_count, slot_count, multi_slot=True)

        self.save_database(db, 'cmstream', output_dir)

        results['cmstream'] = {
            'database': db,
            'stats': stats,
            'word_slot_count': word_slot_count,
            'slot_count': slot_count
        }

        logger.info("DIRTAR processing complete")
        return results


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='DIRTAR: Discovery of Inference Rules from Text for Action Recognition')
    parser.add_argument('corpus_file', help='Path to corpus file')
    parser.add_argument('--min-freq', type=int, default=5, help='Minimum frequency threshold')
    parser.add_argument('--output-dir', type=Path, default=Path('.'), help='Output directory for databases')

    args = parser.parse_args()

    # Create processor
    processor = DIRTARProcessor(min_freq=args.min_freq)

    # Process corpus
    results = processor.process_corpus(args.corpus_file, args.output_dir)

    # Print summary
    print("\\nProcessing Summary:")
    for stream_name, data in results.items():
        stats = data['stats']
        print(f"{stream_name}: {stats[0]} -> {stats[1]} paths, {stats[2]} -> {stats[3]} instances")


if __name__ == '__main__':
    main()