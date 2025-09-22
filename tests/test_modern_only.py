"""
Simple tests for the modern DIRTAR implementation (standalone)
"""

import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dirtar.modern_dirtar import DIRTARProcessor, DependencyInfo, Entry, WordNetHelper


def test_wordnet_helper():
    helper = WordNetHelper()
    assert helper.person_synset is not None

    # Test person replacement
    result = helper.replace_word("john", "PERSON")
    assert result == "person"


def test_dependency_info():
    dep_info = DependencyInfo("man", "nsubj", "PERSON")
    assert dep_info.noun == "man"
    assert dep_info.dep == "nsubj"
    assert dep_info.ner == "PERSON"


def test_entry():
    entry = Entry("walk", "X", "man")
    assert entry.path == "walk"
    assert entry.slot == "X"
    assert entry.word == "man"
    assert entry.count == 1
    assert entry.mi is None

    entry.update_count()
    assert entry.count == 2


def test_dirtar_processor_init():
    processor = DIRTARProcessor(min_freq=3)
    assert processor.min_freq == 3
    assert len(processor.streams) == 6


def test_clean_line():
    processor = DIRTARProcessor()
    result = processor.clean_line("  hello   world  ")
    assert result == "hello world "


def test_decide_swap():
    processor = DIRTARProcessor()

    # No swap case
    result = processor.decide_swap("man", "gun", "nsubj", "dobj")
    assert result == ("man", "gun")

    # Reversible left
    result = processor.decide_swap("man", "gun", "nsubjpass", "dobj")
    assert result == ("gun", "man")

    # Reversible right
    result = processor.decide_swap("man", "gun", "nsubj", "nmod:by")
    assert result == ("gun", "man")


def test_load_basic_triple():
    processor = DIRTARProcessor()
    path_db = {}
    processor._load_basic_triple(path_db, "man", "gun", "shoot")

    assert "X" in path_db
    assert "Y" in path_db
    assert "man" in path_db["X"]
    assert "gun" in path_db["Y"]
    assert path_db["X"]["man"].count == 1
    assert path_db["Y"]["gun"].count == 1


def test_process_line():
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


def test_read_corpus():
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


def test_mutual_information():
    processor = DIRTARProcessor()

    # Test with zero counts (should return 0.0)
    db = {"shoot": {"X": {"man": Entry("shoot", "X", "man", count=0)}}}
    word_slot_count = {}
    slot_count = {}

    mi = processor._mutual_information(db, word_slot_count, slot_count, "shoot", "X", "man")
    assert mi == 0.0


if __name__ == "__main__":
    # Run tests manually
    test_functions = [
        test_wordnet_helper,
        test_dependency_info,
        test_entry,
        test_dirtar_processor_init,
        test_clean_line,
        test_decide_swap,
        test_load_basic_triple,
        test_process_line,
        test_read_corpus,
        test_mutual_information,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")

    if failed == 0:
        print("All tests passed!")
    else:
        print(f"{failed} tests failed.")
        sys.exit(1)