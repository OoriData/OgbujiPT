'''
pytest test

or

pytest test/test_airoboros_obedient_prompting.py
'''
# import pytest

from ogbujipt.prompting.basic import format
from ogbujipt.prompting.model_style import AIROBOROS_OBEDIENT_DELIMITERS

def test_airoboros_obedient_inputs_prompt_building(PROMPTING_USER_QUERY, PROMPTING_CONTEXT_LIST):
    EXPECTED_PROMPT = 'BEGININPUT\nBEGINCONTEXT\nauthor: Isaac Newton\ntitle: Principia Mathematica Volume 1\ndate: 1910\nISBN-13: 978-0521626063\nENDCONTEXT\nAll mathematical propositions\' are of the form S is P\'; S\' is the subject, P\' the predicate.\nENDINPUT\nBEGININPUT\nBEGINCONTEXT\nauthor: Isaac Newton\ntitle: Principia Mathematica Volume 2\ndate: 1912\nISBN-13: 978-0521626070\nENDCONTEXT\nIt is desirable to find some convenient phrase which shall denote the whole collection of values of a function, or the whole collection of entities of a given type.\nENDINPUT\nBEGININPUT\nBEGINCONTEXT\nauthor: Isaac Newton\ntitle: Principia Mathematica Volume 3\ndate: 1913\nISBN-13: 978-0521626087\nENDCONTEXT\nIn this volume we shall define the arithmetical relations between finite cardinal numbers and also their addition and multiplication.\nENDINPUT\nBEGININPUT\nBEGINCONTEXT\nauthor: Isaac Newton\ntitle: Principia Mathematica Volume 4\ndate: 1915\nISBN-13: 978-6969696969\nENDCONTEXT\nFinally, we can get onto the real purpose of this work; to explain why 9 + 10 = 21\nENDINPUT\nBEGININSTRUCTION\nwhat\'s nine plus ten?\nENDINSTRUCTION\n'  # noqa: E501

    prompt = format(
        PROMPTING_USER_QUERY,
        contexts=PROMPTING_CONTEXT_LIST,
        delimiters=AIROBOROS_OBEDIENT_DELIMITERS,
        context_with_metadata=True,
        )
    assert prompt == EXPECTED_PROMPT