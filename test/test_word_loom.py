'''
pytest test

or

pytest test/test_text_splitter.py
'''
# ruff: noqa: E501

import os
import sys
# import pkgutil

import pytest

from ogbujipt import word_loom

# Following only works to get the content as octets
# data = pkgutil.get_data('ogbujipt', 'resources/wordloom/sample.toml')


@pytest.fixture
def SAMPLE_TOML_STR():
    # Load language material (word loom) format
    fpath = os.path.dirname(sys.modules['ogbujipt'].__file__)

    with open(os.path.join(fpath, 'resources/wordloom/sample.toml'), 'rb') as fp:
        toml_content = fp.read()
    return toml_content


@pytest.fixture
def SAMPLE_TOML_FP():
    # Load language material (word loom) format
    fpath = os.path.dirname(sys.modules['ogbujipt'].__file__)

    return open(os.path.join(fpath, 'resources/wordloom/sample.toml'), 'rb')

def test_load_fp_vs_str(SAMPLE_TOML_STR, SAMPLE_TOML_FP):
    loom1 = word_loom.load(SAMPLE_TOML_STR)
    loom2 = word_loom.load(SAMPLE_TOML_FP)
    SAMPLE_TOML_FP.close()
    assert loom1 == loom2


def test_sample_texts_check(SAMPLE_TOML_STR):
    # print(SAMPLE_TOML)
    loom = word_loom.load(SAMPLE_TOML_STR)
    assert list(sorted(loom.keys())) == ['davinci3_instruct_system', 'i18n_context', 'write_i18n_advocacy']
    assert list(sorted([v[:20] for v in loom.values()])) == ['Internationalization', 'Obey the instruction', '{davinci3_instruct_s']
    assert [v.markers or [] for v in loom.values()] == [[], [], ['davinci3_instruct_system', 'i18n_context']]
    assert loom['davinci3_instruct_system'].lang == 'en'

    # Default language is English
    loom1 = word_loom.load(SAMPLE_TOML_STR, lang='en')
    assert loom1 == loom

    loom = word_loom.load(SAMPLE_TOML_STR, lang='fr')
    assert list(sorted(loom.keys())) == ['hardcoded_food', 'translate_request']
    assert list(sorted([v[:20] for v in loom.values()])) == ['Comment dit-on en an', 'pomme de terre']
    assert [v.markers or [] for v in loom.values()] == [['hardcoded_food'], []]
    assert loom['hardcoded_food'].lang == 'fr'
