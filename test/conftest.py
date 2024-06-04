'''
Common tools & resources for test cases
'''
# ruff: noqa: E501

import pytest
from openai.types.completion import Completion, CompletionChoice, CompletionUsage
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChatCompletionMessage


@pytest.fixture
def PROMPTING_USER_QUERY():
    return 'what\'s nine plus ten?'


@pytest.fixture
def PROMPTING_CONTEXT():
    return 'The entirety of "Principia Mathematica" by Isaac Newton'


@pytest.fixture
def PROMPTING_CONTEXT_LIST():
    return [
        (
            'author: Isaac Newton\n' 'title: Principia Mathematica Volume 1\n' 'date: 1910\n' 'ISBN-13: 978-0521626063',
            'All mathematical propositions\' are of the form S is P\'; S\' is the subject, P\' the predicate.',
        ),
        (
            'author: Isaac Newton\n' 'title: Principia Mathematica Volume 2\n' 'date: 1912\n' 'ISBN-13: 978-0521626070',
            'It is desirable to find some convenient phrase which shall denote the whole collection of values of a function, or the whole collection of entities of a given type.',
        ),
        (
            'author: Isaac Newton\n' 'title: Principia Mathematica Volume 3\n' 'date: 1913\n' 'ISBN-13: 978-0521626087',
            'In this volume we shall define the arithmetical relations between finite cardinal numbers and also their addition and multiplication.',
        ),
        (
            'author: Isaac Newton\n' 'title: Principia Mathematica Volume 4\n' 'date: 1915\n' 'ISBN-13: 978-6969696969',
            'Finally, we can get onto the real purpose of this work; to explain why 9 + 10 = 21',
        ),
    ]


@pytest.fixture
def BAD_XML_CODE():
    return '''\
<earth>
<country><b>Russia</country></b>
<capital>Moscow</capital>
</Earth>'''


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
def COME_THUNDER_POEM_CHUNKS():
    return [
        'Now that the triumphant march has entered the last street corners,\nRemember, O dancers, the thunder among the clouds…\n\nNow that the laughter, broken in two, hangs tremulous between the teeth,\nRemember, O Dancers, the lightning beyond the earth…',
        'Now that the triumphant march has entered the last street corners,\nRemember, O dancers, the thunder among the clouds…\n\nNow that the laughter, broken in two, hangs tremulous between the teeth,\nRemember, O Dancers, the lightning beyond the earth…\n\nThe smell of blood already floats in the lavender-mist of the afternoon.\nThe death sentence lies in ambush along the corridors of power;\nAnd a great fearful thing already tugs at the cables of the open air,\nA nebula immense and immeasurable, a night of deep waters —\nAn iron dream unnamed and unprintable, a path of stone.',
        'Now that the laughter, broken in two, hangs tremulous between the teeth,\nRemember, O Dancers, the lightning beyond the earth…\n\nThe smell of blood already floats in the lavender-mist of the afternoon.\nThe death sentence lies in ambush along the corridors of power;\nAnd a great fearful thing already tugs at the cables of the open air,\nA nebula immense and immeasurable, a night of deep waters —\nAn iron dream unnamed and unprintable, a path of stone.\n\nThe drowsy heads of the pods in barren farmlands witness it,\nThe homesteads abandoned in this century’s brush fire witness it:\nThe myriad eyes of deserted corn cobs in burning barns witness it:\nMagic birds with the miracle of lightning flash on their feathers…',
        'The smell of blood already floats in the lavender-mist of the afternoon.\nThe death sentence lies in ambush along the corridors of power;\nAnd a great fearful thing already tugs at the cables of the open air,\nA nebula immense and immeasurable, a night of deep waters —\nAn iron dream unnamed and unprintable, a path of stone.\n\nThe drowsy heads of the pods in barren farmlands witness it,\nThe homesteads abandoned in this century’s brush fire witness it:\nThe myriad eyes of deserted corn cobs in burning barns witness it:\nMagic birds with the miracle of lightning flash on their feathers…\n\nThe arrows of God tremble at the gates of light,\nThe drums of curfew pander to a dance of death;',
    ]


@pytest.fixture
def BASIC_EVEN_BLOCK():
    # Simple example of text with even-length lines
    return '''\
0123456789
abcdefghij
ABCDEFGHIJ
klmnopqrst'''


@pytest.fixture
def LOREM_IPSUM():
    # Lorem ipsum text
    return (
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce vestibulum nisi eget mauris malesuada, '
        'quis facilisis arcu vehicula. Sed consequat, quam ut auctor volutpat, augue ex tincidunt massa, in varius '
        'nulla ex vel ipsum. Nullam vitae eros nec ante sagittis luctus. Nullam scelerisque dolor eu orci iaculis, '
        'at convallis nulla luctus. Praesent eget ex id arcu facilisis varius vel id neque. Donec non orci eget '
        'elit aliquam tempus. Sed at tortor at tortor congue dictum. Nulla varius erat at libero lacinia, id '
        'dignissim risus auctor. Ut eu odio vehicula, tincidunt justo ac, viverra erat. Sed nec sem sit amet erat '
        'malesuada finibus. Nulla sit amet diam nec dolor tristique dignissim. Sed vehicula, justo nec posuere '
        'eleifend, libero ligula interdum neque, at lacinia arcu quam non est. Integer aliquet, erat id dictum '
        'euismod, felis libero blandit lorem, nec ullamcorper quam justo at elit.'
    )


@pytest.fixture
def OPENAI_TEXT_RESPONSE_OBJECT():
    # Example of OpenAI response object
    # Response generated by prompting with "Recite the Fitness Gram Pacer Test script."
    return Completion(
        id='cmpl-13ca5e85-a890-4c70-bbbf-9f4825728f23',
        choices=[
            CompletionChoice(
                finish_reason='length',
                index=0,
                logprobs=None,
                text='…is an exceptional employee who has made significant contributions to our company.',
            )
        ],
        created=1701544442,
        model='.local/share/models/TheBloke_Llama-2-7B-32K-Instruct-GGUF/llama-2-7b-32k-instruct.Q5_K_M.gguf',
        object='text_completion',
        system_fingerprint=None,
        usage=CompletionUsage(completion_tokens=256, prompt_tokens=12, total_tokens=268),
    )


# TODO: Finish up, then enable test in test/test_ogbujipt.py
@pytest.fixture
def OPENAI_MSG_RESPONSE_OBJECT():
    return ChatCompletion(
        id='chatcmpl-923076c2-d1d5-473b-bdc6-6e803037357a',
        choices=[
            Choice(
                finish_reason='length',
                index=0,
                message=ChatCompletionMessage(
                    content='…is an exceptional employee who has made significant contributions to our company.',
                    role='assistant',
                    function_call=None,
                    tool_calls=None,
                ),
                logprobs=None
            )
        ],
        created=1701544162,
        model='.local/share/models/TheBloke_Llama-2-7B-32K-Instruct-GGUF/llama-2-7b-32k-instruct.Q5_K_M.gguf',
        object='chat.completion',
        system_fingerprint=None,
        usage=CompletionUsage(completion_tokens=256, prompt_tokens=22, total_tokens=278),
    )
