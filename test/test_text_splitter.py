'''
pytest test

or

pytest test/test_text_splitter.py
'''
import pytest

from ogbujipt.text_helper import text_splitter

# Also test: chunk_overlap > chunk_size, chunk_overlap == chunk_size, etc.


def test_split_basic():
    # Need to test chunk_size > total length
    # chunks = text_splitter(BASIC_EVEN_BLOCK, chunk_size=100, chunk_overlap=10)
    chunks = text_splitter(
        BASIC_EVEN_BLOCK, chunk_size=21, chunk_overlap=3, separator='\n')
    # import pprint; pprint.pprint(chunks)  # noqa
    assert len(chunks) == 3
    assert chunks == [
        '0123456789\nabcdefghij', '0123456789\nabcdefghij\nABCDEFGHIJ',
        'abcdefghij\nABCDEFGHIJ\nklmnopqrst'
        ]


# @pytest.mark.skip(reason="Still working on it")
def test_split_poem():
    # Just use a subset of the poem
    poem_text = COME_THUNDER.partition('The arrows of God')[0]
    chunks = text_splitter(poem_text, chunk_size=200, chunk_overlap=30)
    # print(chunks[0], '\n---')
    assert chunks[0] == 'Now that the triumphant march has entered the last street corners,\nRemember, O dancers, the thunder among the clouds…\n\nNow that the laughter, broken in two, hangs tremulous between the teeth,\nRemember, O Dancers, the lightning beyond the earth…'
    assert chunks[1] == 'Now that the triumphant march has entered the last street corners,\nRemember, O dancers, the thunder among the clouds…\n\nNow that the laughter, broken in two, hangs tremulous between the teeth,\nRemember, O Dancers, the lightning beyond the earth…\n\nThe smell of blood already floats in the lavender-mist of the afternoon.\nThe death sentence lies in ambush along the corridors of power;\nAnd a great fearful thing already tugs at the cables of the open air,\nA nebula immense and immeasurable, a night of deep waters —\nAn iron dream unnamed and unprintable, a path of stone.'
    assert chunks[2] == 'Now that the laughter, broken in two, hangs tremulous between the teeth,\nRemember, O Dancers, the lightning beyond the earth…\n\nThe smell of blood already floats in the lavender-mist of the afternoon.\nThe death sentence lies in ambush along the corridors of power;\nAnd a great fearful thing already tugs at the cables of the open air,\nA nebula immense and immeasurable, a night of deep waters —\nAn iron dream unnamed and unprintable, a path of stone.\n\nThe drowsy heads of the pods in barren farmlands witness it,\nThe homesteads abandoned in this century’s brush fire witness it:\nThe myriad eyes of deserted corn cobs in burning barns witness it:\nMagic birds with the miracle of lightning flash on their feathers…'
    # import pprint; pprint.pprint(chunks)  # noqa
    assert len(chunks) == 3


# One of Christopher Okigbo's greatest poems
COME_THUNDER = '''\
Now that the triumphant march has entered the last street corners,
Remember, O dancers, the thunder among the clouds…

Now that the laughter, broken in two, hangs tremulous between the teeth,
Remember, O Dancers, the lightning beyond the earth…

The smell of blood already floats in the lavender-mist of the afternoon.
The death sentence lies in ambush along the corridors of power;
And a great fearful thing already tugs at the cables of the open air,
A nebula immense and immeasurable, a night of deep waters —
An iron dream unnamed and unprintable, a path of stone.

The drowsy heads of the pods in barren farmlands witness it,
The homesteads abandoned in this century’s brush fire witness it:
The myriad eyes of deserted corn cobs in burning barns witness it:
Magic birds with the miracle of lightning flash on their feathers…

The arrows of God tremble at the gates of light,
The drums of curfew pander to a dance of death;

And the secret thing in its heaving
Threatens with iron mask
The last lighted torch of the century…'''


BASIC_EVEN_BLOCK = '''\
0123456789
abcdefghij
ABCDEFGHIJ
klmnopqrst
'''
