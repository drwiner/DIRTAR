"""
Tests for the modern DIRTAR implementation
"""

import pytest
import tempfile
from pathlib import Path
from collections import defaultdict

from dirtar.modern_dirtar import DIRTARProcessor, DependencyInfo, Entry, WordNetHelper


class TestWordNetHelper:
    def test_init(self):
        helper = WordNetHelper()
        assert helper.person_synset is not None

    def test_replace_word_person(self):
        helper = WordNetHelper()
        result = helper.replace_word("john", "PERSON")
        assert result == "person"

    def test_replace_word_unknown(self):
        helper = WordNetHelper()
        result = helper.replace_word("asdfghjkl", "O")
        # Should return person synset lemma for unknown words
        assert result == "person"


class TestDependencyInfo:
    def test_creation(self):
        dep_info = DependencyInfo("man", "nsubj", "PERSON")
        assert dep_info.noun == "man"
        assert dep_info.dep == "nsubj"
        assert dep_info.ner == "PERSON"


class TestEntry:
    def test_creation(self):
        entry = Entry("walk", "X", "man")
        assert entry.path == "walk"
        assert entry.slot == "X"
        assert entry.word == "man"
        assert entry.count == 1
        assert entry.mi is None

    def test_update_count(self):
        entry = Entry("walk", "X", "man")
        entry.update_count()
        assert entry.count == 2


class TestDIRTARProcessor:
    def test_init(self):
        processor = DIRTARProcessor(min_freq=3)
        assert processor.min_freq == 3
        assert len(processor.streams) == 6

    def test_clean_line(self):
        processor = DIRTARProcessor()
        result = processor.clean_line("  hello   world  ")
        assert result == "hello world "

    def test_decide_swap_no_swap(self):
        processor = DIRTARProcessor()
        result = processor.decide_swap("man", "gun", "nsubj", "dobj")
        assert result == ("man", "gun")

    def test_decide_swap_reversible_left(self):
        processor = DIRTARProcessor()
        result = processor.decide_swap("man", "gun", "nsubjpass", "dobj")
        assert result == ("gun", "man")

    def test_decide_swap_reversible_right(self):
        processor = DIRTARProcessor()
        result = processor.decide_swap("man", "gun", "nsubj", "nmod:by")
        assert result == ("gun", "man")

    def test_fix_malformed_line_comma(self):
        processor = DIRTARProcessor()
        entries = ["(", "word", "verb", "object"]
        result = processor._fix_malformed_line(entries)
        assert result == ["(comma word", "verb", "object"]

    def test_fix_malformed_line_short(self):
        processor = DIRTARProcessor()
        entries = ["subject", "verb"]
        result = processor._fix_malformed_line(entries)
        assert result == ["subject", "verb", "(number - NP - NUMBER - dobj)"]

    def test_load_basic_triple(self):
        processor = DIRTARProcessor()
        path_db = {}
        processor._load_basic_triple(path_db, "man", "gun", "shoot")

        assert "X" in path_db
        assert "Y" in path_db
        assert "man" in path_db["X"]
        assert "gun" in path_db["Y"]
        assert path_db["X"]["man"].count == 1
        assert path_db["Y"]["gun"].count == 1

    def test_load_multislot_triple(self):
        processor = DIRTARProcessor()
        path_db = {}
        x_info = DependencyInfo("man", "nsubj", "PERSON")
        y_info = DependencyInfo("gun", "dobj", "O")

        processor._load_multislot_triple(path_db, x_info, y_info, "shoot")

        assert "x_nsubj" in path_db
        assert "y_dobj" in path_db
        assert "man" in path_db["x_nsubj"]
        assert "gun" in path_db["y_dobj"]

    def test_mutual_information_zero_counts(self):
        processor = DIRTARProcessor()
        db = {"shoot": {"X": {"man": Entry("shoot", "X", "man", count=0)}}}
        word_slot_count = {}
        slot_count = {}

        mi = processor._mutual_information(db, word_slot_count, slot_count, "shoot", "X", "man")
        assert mi == 0.0

    def test_apply_frequency_filter(self):
        processor = DIRTARProcessor(min_freq=2)
        # Add some test data
        processor.streams['tstream'] = [
            processor.Triple("man", "shoot", "gun"),
            processor.Triple("man", "shoot", "rifle"),
            processor.Triple("woman", "walk", "street"),  # This path appears only once
            processor.Triple("person", "shoot", "target"),
        ]

        stats = processor.apply_frequency_filter('tstream')
        unfiltered_paths, filtered_paths, unfiltered_instances, filtered_instances, instances = stats

        # Should have 2 distinct paths originally, 1 after filtering (only "shoot" appears >= 2 times)
        assert unfiltered_paths == 2
        assert filtered_paths == 1
        assert unfiltered_instances == 4
        assert filtered_instances == 3  # The 3 "shoot" instances

    def test_process_line_integration(self):
        processor = DIRTARProcessor()
        test_line = "(man - NNP - PERSON - nsubj), shoot, (gun - NN - O - dobj)"

        initial_lengths = {k: len(v) for k, v in processor.streams.items()}
        processor._process_line(test_line)

        # Check that streams were populated
        for stream_name in processor.streams:
            assert len(processor.streams[stream_name]) > initial_lengths[stream_name]

        # Check basic stream
        assert len(processor.streams['tstream']) == 1
        triple = processor.streams['tstream'][0]
        assert triple.X == "man"
        assert triple.path == "shoot"
        assert triple.Y == "gun"

    def test_read_corpus_with_temp_file(self):
        processor = DIRTARProcessor()

        # Create a temporary file with test data
        test_data = """(man - NNP - PERSON - nsubj), shoot, (gun - NN - O - dobj)
(woman - NNP - PERSON - nsubj), walk, (street - NN - O - dobj)
(person - NN - PERSON - nsubj), aim, (target - NN - O - dobj)"""

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(test_data)
            temp_path = f.name

        try:
            processor.read_corpus(temp_path)

            # Check that streams were populated
            assert len(processor.streams['tstream']) == 3
            assert len(processor.streams['wstream']) == 3
            assert len(processor.streams['cmstream']) == 3

            # Check specific content
            tstream_paths = [t.path for t in processor.streams['tstream']]
            assert "shoot" in tstream_paths
            assert "walk" in tstream_paths
            assert "aim" in tstream_paths

        finally:
            Path(temp_path).unlink()

    def test_save_database(self):
        processor = DIRTARProcessor()
        test_db = {"test": {"X": {"word": Entry("test", "X", "word")}}}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            processor.save_database(test_db, "test", output_dir)

            # Check that file was created
            expected_file = output_dir / "dirtar_database_test.pkl"
            assert expected_file.exists()

            # Try to load it back
            import pickle
            with open(expected_file, 'rb') as f:
                loaded_db = pickle.load(f)

            assert "test" in loaded_db
            assert "X" in loaded_db["test"]
            assert "word" in loaded_db["test"]["X"]


if __name__ == "__main__":
    pytest.main([__file__])