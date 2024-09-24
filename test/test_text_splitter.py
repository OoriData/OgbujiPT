# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/test_text_splitter.py
'''
pytest test

or

pytest test/test_text_splitter.py
'''
# ruff: noqa: E501

import pytest

from ogbujipt.text_helper import text_split, text_split_fuzzy

# Also test: chunk_overlap > chunk_size, chunk_overlap == chunk_size, etc.


def test_split_basic(BASIC_EVEN_BLOCK):
    # Need to test chunk_size > total length
    # chunks = text_split_fuzzy(BASIC_EVEN_BLOCK, chunk_size=100, chunk_overlap=10)
    chunks = list(text_split_fuzzy(
        BASIC_EVEN_BLOCK, chunk_size=21, chunk_overlap=3, separator='\n'))
    # import pprint; pprint.pprint(chunks)  # noqa
    assert len(chunks) == 3
    assert chunks == [
        '0123456789\nabcdefghij', '0123456789\nabcdefghij\nABCDEFGHIJ',
        'abcdefghij\nABCDEFGHIJ\nklmnopqrst'
        ]


# @pytest.mark.skip(reason="Still working on it")
def test_split_poem(COME_THUNDER_POEM):
    # Just use a subset of the poem
    poem_text = COME_THUNDER_POEM.partition('The arrows of God')[0]
    chunks = list(text_split_fuzzy(poem_text, chunk_size=200, chunk_overlap=30))
    # print(chunks[0], '\n---')
    assert chunks[0] == 'Now that the triumphant march has entered the last street corners,\nRemember, O dancers, the thunder among the clouds…\n\nNow that the laughter, broken in two, hangs tremulous between the teeth,\nRemember, O Dancers, the lightning beyond the earth…'
    assert chunks[1] == 'Now that the triumphant march has entered the last street corners,\nRemember, O dancers, the thunder among the clouds…\n\nNow that the laughter, broken in two, hangs tremulous between the teeth,\nRemember, O Dancers, the lightning beyond the earth…\n\nThe smell of blood already floats in the lavender-mist of the afternoon.\nThe death sentence lies in ambush along the corridors of power;\nAnd a great fearful thing already tugs at the cables of the open air,\nA nebula immense and immeasurable, a night of deep waters —\nAn iron dream unnamed and unprintable, a path of stone.'
    assert chunks[2] == 'Now that the laughter, broken in two, hangs tremulous between the teeth,\nRemember, O Dancers, the lightning beyond the earth…\n\nThe smell of blood already floats in the lavender-mist of the afternoon.\nThe death sentence lies in ambush along the corridors of power;\nAnd a great fearful thing already tugs at the cables of the open air,\nA nebula immense and immeasurable, a night of deep waters —\nAn iron dream unnamed and unprintable, a path of stone.\n\nThe drowsy heads of the pods in barren farmlands witness it,\nThe homesteads abandoned in this century’s brush fire witness it:\nThe myriad eyes of deserted corn cobs in burning barns witness it:\nMagic birds with the miracle of lightning flash on their feathers…'
    # import pprint; pprint.pprint(chunks)  # noqa
    assert len(chunks) == 3


def test_separator_not_found(BASIC_EVEN_BLOCK):
    with pytest.warns(UserWarning):
        chunks = list(text_split_fuzzy(
            BASIC_EVEN_BLOCK, chunk_size=21, chunk_overlap=3, separator='!!!'))
    assert len(chunks) == 1
    assert len(chunks[0]) == len(BASIC_EVEN_BLOCK)


def test_separator_chunk_size_too_large(BASIC_EVEN_BLOCK):
    # Just use a subset of the poem
    with pytest.warns(UserWarning):
        chunks = list(text_split_fuzzy(
            BASIC_EVEN_BLOCK, chunk_size=100, chunk_overlap=10, separator='!!!'))
    assert len(chunks) == 1
    assert len(chunks[0]) == len(BASIC_EVEN_BLOCK)

def test_zero_overlap(LOREM_IPSUM):
    chunks = list(text_split(LOREM_IPSUM, chunk_size=100, separator=' '))
    assert len(chunks) == 10
    assert chunks[0] == 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce vestibulum nisi eget mauris'
    assert chunks[2] == 'massa, in varius nulla ex vel ipsum. Nullam vitae eros nec ante sagittis luctus. Nullam scelerisque'
    assert chunks[3] == 'dolor eu orci iaculis, at convallis nulla luctus. Praesent eget ex id arcu facilisis varius vel id'
    assert chunks[-1] == 'quam justo at elit.'
    for chunk in chunks:
        assert len(chunk) <= 100


# TODO: Markup split based on below:

'''
# Hello

Goodbye

## World

Neighborhood

### Spam

Spam spam spam!

# Eggs

Green, with ham
'''

# Check the differences with e.g.
# list(text_split(s, chunk_size=50, separator=r'^(#)'))
# Where chunk_size varies from 5 to 100 & sep is also e.g. r'^#', r'^(#+)', etc.

# Notice, 

# Based on https://regex101.com/r/cVCCSg/1 This wacky example:

'''
import re
text= """# Heading 1
## heading 2 (some text in parentheses)
###Heading 3

Don't match the following:

[Some internal link](
#foo)
[Some internal link](
#foo)
[Some internal link](
#foo
)"""
print( re.sub(r'(\[[^][]*]\([^()]*\))|^(#+)(.*)', lambda x: x.group(1) if x.group(1) else "<h{1}>{0}</h{1}>".format(x.group(3), len(x.group(2))), text, flags=re.M) )
'''

if __name__ == '__main__':
    raise SystemExit("Attention! Run with pytest")
