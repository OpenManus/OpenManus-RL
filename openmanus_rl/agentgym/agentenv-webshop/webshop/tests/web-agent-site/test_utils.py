import logging
import pytest
import random
import shutil
from pathlib import Path
from web_agent_site.utils import *

def test_random_idx():
    random.seed(24)
    weights = [random.randint(0, 10) for _ in range(0, 50)]
    cml_weights = [0]
    for w in weights:
        cml_weights.append(cml_weights[-1] + w)
    idx_1, expected_1 = random_idx(cml_weights), 44
    idx_2, expected_2 = random_idx(cml_weights), 15
    idx_3, expected_3 = random_idx(cml_weights), 36
    assert idx_1 == expected_1
    assert idx_2 == expected_2
    assert idx_3 == expected_3

def test_setup_logger():
    LOG_DIR = 'user_session_logs_test/'
    user_log_dir = Path(LOG_DIR)
    user_log_dir.mkdir(parents=True, exist_ok=True)
    session_id = "ABC"

    logger = setup_logger(session_id, user_log_dir)
    log_file = Path(LOG_DIR + "/" + session_id + ".jsonl")
    assert Path(log_file).is_file()
    assert logger.level == logging.INFO

    content = "Hello there"
    logger.info(content)
    assert log_file.read_text().strip("\n") == content

    shutil.rmtree(LOG_DIR)

def test_generate_mturk_code():
    suite = [
        ('', 'DA39A3EE5E'),
        ('ABC', '3C01BDBB26'),
        ('123', '40BD001563'),
        ('1A1', '10E7DB0A44'),
        ('$%^ABC', '5D5607D24E')
    ]
    for session_id, expected in suite:
        output = generate_mturk_code(session_id)
        assert type(expected) is str
        assert output == expected