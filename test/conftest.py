'''
Common tools & resources for test cases
'''

import pytest


@pytest.fixture
def COME_THUNDER_POEM():
    # One of Christopher Okigbo's greatest poems
    return '''\
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


@pytest.fixture
def BASIC_EVEN_BLOCK():
    # Simple example of text with even-length lines
    return '''\
0123456789
abcdefghij
ABCDEFGHIJ
klmnopqrst'''
