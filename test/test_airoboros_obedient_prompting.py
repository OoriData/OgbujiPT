'''
pytest test

or

pytest test/test_airoboros_obedient_prompting.py
'''

from ogbujipt.prompting.basic import format
from ogbujipt.prompting.model_style import concat_input_prompts, AIROBOROS_OBEDIENT_DELIMITERS


def test_basic_prompt():
    EXPECTED_PROMPT = 'USER:\nBEGININPUT\nBEGINCONTEXT\nauthor: John Doe\ntitle: The Art of Baking Bread\npublication: Food Lovers Magazine\ndate: 2022-01-15\nurl: foodloversmagazine.com/the-art-of-baking-bread\nENDCONTEXT\nIn this article, we will explore the art of baking bread at home.\nBEGINCONTEXT\nauthor: Jane Smith\ntitle: A Day in the Life of a Professional Baker\nblog: Behind the Bakery Counter\ndate: 2021-12-20\nurl: behindthebakerycounter.com/a-day-in-the-life-of-a-professional-baker\nENDCONTEXT\nAs a professional baker, my day starts early - usually around 4 am.\nBEGINCONTEXT\nauthor: Emily Johnson\ntitle: The History of Bread Making\ndate: 2022-01-10\npublication: Culinary Chronicles\nurl: culinarychronicles.com/the-history-of-bread-making\nENDCONTEXT\nBread has been a staple food for thousands of years, dating back to ancient civilizations in Egypt, Mesopotamia, and Greece.\n\nENDINPUT\n\nBEGININSTRUCTION\nHow do I make sourdough bread? What is its cultural significance?\n\n[citations]\nENDINSTRUCTION\nASSISTANT:'  # noqa

    ctx_texts = []

    # Taken from https://huggingface.co/datasets/jondurbin/airoboros-gpt4/blob/main/full-example.md
    # rip PEP8 line length (not always mourned)
    ctx_texts.append(('''author: John Doe
title: The Art of Baking Bread
publication: Food Lovers Magazine
date: 2022-01-15
url: foodloversmagazine.com/the-art-of-baking-bread''',
        'In this article, we will explore the art of baking bread at home.'))  # noqa

    ctx_texts.append(('''author: Jane Smith
title: A Day in the Life of a Professional Baker
blog: Behind the Bakery Counter
date: 2021-12-20
url: behindthebakerycounter.com/a-day-in-the-life-of-a-professional-baker''',
        'As a professional baker, my day starts early - usually around 4 am.'))  # noqa

    ctx_texts.append(('''author: Emily Johnson
title: The History of Bread Making
date: 2022-01-10
publication: Culinary Chronicles
url: culinarychronicles.com/the-history-of-bread-making''',
        'Bread has been a staple food for thousands of years, '  # noqa
        'dating back to ancient civilizations in Egypt, Mesopotamia, '  # noqa
        'and Greece.'))  # noqa

    given_inputs = concat_input_prompts(ctx_texts)

    prompt = format(
        'How do I make sourdough bread? What is its cultural significance?\n\n[citations]',  # noqa
        contexts=given_inputs,
        delimiters=AIROBOROS_OBEDIENT_DELIMITERS
        )
    print(prompt)
    print(repr(prompt))

    assert prompt == EXPECTED_PROMPT
