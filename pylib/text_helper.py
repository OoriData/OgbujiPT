# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.text_helper

'''
Routines to help with text processing
'''
import re
import warnings
from itertools import zip_longest


# XXX: Do we want a case-insensitive separator regex flag?
def text_split(text: str, chunk_size: int, separator: str='\n\n', joiner=None, len_func=len):
    '''
    Split string and generate the sequence of chunks

    >>> from ogbujipt.text_helper import text_split
    >>> list(text_split('She sells seashells by the seashore', chunk_size=5, separator=' '))
    ['She', 'sells', 'seashells', 'by', 'the', 'seashore']
    >>> list(text_split('She sells seashells by the seashore', chunk_size=10, separator=' '))
    ['She sells', 'seashells', 'by the', 'seashore']
    # Notice case sensitivity, plus the fact that the separator is not included in the chunks
    >>> list(text_split('She sells seashells by the seashore', chunk_size=10, separator='s'))
    ['She ', 'ells ', 'ea', 'hell', ' by the ', 'ea', 'hore']
    >>> list(text_split('She\tsells seashells\tby the seashore', chunk_size=10, separator='\\s'))
    ['She\tsells', 'seashells', 'by the', 'seashore']
    >>> list(text_split('She\tsells seashells\tby the seashore', chunk_size=10, separator='\\s', joiner=' '))
    ['She sells', 'seashells', 'by the', 'seashore']

    Args:
        text (str): String to be split into chunks

        chunk_size (int): Guidance on maximum length (based on distance_function) of each chunk

        seperator (str, optional): Regex used to split `text` into sections. Do not include outer capturing parenthesis.
            Don't forget to use escaping where necessary.

        joiner (str, optional): Exact string used to rejoin any sections in order to meet target length
            defaults to using the literal match from the separator

        len_func (callable, optional): Function to measure chunk length, len() by default

    Returns:
        chunks (List[str]): List of chunks of the text provided
    '''
    assert separator, 'Separator must be non-empty'
    
    if ((not isinstance(text, str))
        or (not isinstance(separator, str))):
        raise ValueError(f'text and separator must be strings.\n'
                         f'Got {text.__class__} for text and {separator.__class__} for separator')
    
    if ((not isinstance(chunk_size, int)) or (chunk_size <= 0)):
        raise ValueError(f'chunk_size must be a positive integer, got {chunk_size}.')
    
    # Split the text by the separator
    if joiner is None:
        separator = f'({separator})'
    sep_pat = re.compile(separator)
    raw_split = re.split(sep_pat, text)

    # Rapid aid to understanding following logic:
    # data = ['a',' ','b','\t','c']
    # list(zip_longest(data[0::2], data[1::2], fillvalue=''))
    #     →[('a', ' '), ('b', '\t'), ('c', '')]
    fine_split = ([ i for i in zip_longest(raw_split[0::2], raw_split[1::2], fillvalue='') ]
                    if joiner is None else re.split(sep_pat, text))

    if len(fine_split) <= 1:
        warnings.warn(f'No splits detected. Perhaps a problem with separator? ({repr(separator)})?')

    curr_chunk = []
    chunk_len = 0

    for fs in fine_split:
        (fs, sep) = fs if joiner is None else (fs, joiner)
        if not fs: continue  # noqa E701
        sep_len = len_func(sep)
        len_fs = len_func(fs)
        # if len_fs > chunk_size:
        #     warnings.warn(f'One of the splits is larger than the chunk size. '
        #                   f'Consider increasing the chunk size or splitting the text differently.')

        if chunk_len + len_fs > chunk_size:
            chunk = ''.join(curr_chunk[:-1])
            if chunk: yield chunk  # noqa E701
            curr_chunk, chunk_len = [fs, sep], len_fs + sep_len
        else:
            curr_chunk.extend((fs, sep))
            chunk_len += len_fs + sep_len

    if curr_chunk:
        chunk = ''.join(curr_chunk[:-1])
        if chunk: yield chunk  # noqa E701


def text_split_fuzzy(text: str,
        chunk_size: int,
        chunk_overlap: int=0,
        separator: str='\n\n',
        joiner=None,
        len_func=len
    ):
    '''
    Split string in a "fuzzy" manner and generate the sequence of chunks
    Will generally not split the text into sequences of chunk_size length, but will instead preserve some overlap
    on either side of the given separators.
    This results in slightly larger chunks than the chunk_size number by itself suggests.

    >>> from ogbujipt.text_helper import text_split_fuzzy
    >>> chunks = list(text_split_fuzzy('she sells seashells by the seashore', chunk_size=5, separator=' '))

    Args:
        text (str): (Multiline) string to be split into chunks

        chunk_size (int): Target length (as determined by len_func) per chunk

        chunk_overlap (int, optional): Number of characters to overlap at the edges of chunks

        seperator (str, optional): Regex used to split `text` into sections. Do not include outer capturing parenthesis.
            Don't forget to use escaping where necessary.

        joiner (str, optional): Exact string used to rejoin any sections in order to meet target length
            defaults to using the literal match from the separator

        len_func (callable, optional): Function to measure chunk length, len() by default

    Yields:
        Sequence of chunks (str) from splitting
    '''
    assert separator, 'Separator must be non-empty'
    
    if ((not isinstance(text, str))
        or (not isinstance(separator, str))):
        raise ValueError(f'text and separator must be strings.\n'
                         f'Got {text.__class__} for text and {separator.__class__} for separator')
    
    if chunk_overlap == 0:
        msg = 'chunk_overlap must be a positive integer. For no overlap, use text_split() instead'
        raise ValueError(msg)

    if ((not isinstance(chunk_size, int))
        or (not isinstance(chunk_overlap, int))
        or (chunk_size <= 0)
        or (chunk_overlap < 0)
        or (chunk_size < chunk_overlap)):
        raise ValueError(f'chunk_size must be a positive integer, '
                         f'chunk_overlap must be a non-negative integer, and'
                         f'chunk_size must be greater than chunk_overlap.\n'
                         f'Got {chunk_size} chunk_size and {chunk_overlap} chunk_overlap.')

    # Split up the text by the separator
    # FIXME: Need a step for escaping regex
    sep_pat = re.compile(separator)
    fine_split = re.split(sep_pat, text)
    separator_len = len_func(separator)

    if len(fine_split) <= 1:
        warnings.warn(f'No splits detected. Perhaps a problem with separator? ({repr(separator)})?')

    at_least_one_chunk = False

    # Combine the small pieces into medium size chunks
    # chunks will accumulate processed text chunks as we go along
    # curr_chunk will be a list of subchunks comprising the main, current chunk
    # back_overlap will be appended, once ready, to the end of the previous chunk (if any)
    # fwd_overlap will be prepended, once ready, to the start of the next chunk
    prev_chunk = ''
    curr_chunk, curr_chunk_len = [], 0
    back_overlap, back_overlap_len = None, 0  # None signals not yet gathering
    fwd_overlap, fwd_overlap_len = None, 0

    for s in fine_split:
        if not s: continue  # noqa E701
        split_len = len_func(s) + separator_len
        # Check for full back_overlap (if relevant, i.e. back_overlap isn't None)
        if back_overlap is not None and (back_overlap_len + split_len > chunk_overlap):  # noqa: F821
            prev_chunk.extend(back_overlap)
            back_overlap, back_overlap_len = None, 0

        # Will adding this split take us into overlap room?
        if curr_chunk_len + split_len > (chunk_size - chunk_overlap):
            fwd_overlap, fwd_overlap_len = [], 0  # Start gathering

        # Will adding this split take us over chunk size?
        if curr_chunk_len + split_len > chunk_size:
            # If so, complete current chunk & start a new one

            # fwd_overlap should be non-None at this point, so check empty
            if not fwd_overlap and curr_chunk:
                # If empty, look back to make sure there is some overlap
                fwd_overlap.append(curr_chunk[-1])

            prev_chunk = separator.join(prev_chunk)
            if prev_chunk:
                at_least_one_chunk = True
                yield prev_chunk
            prev_chunk = curr_chunk
            # fwd_overlap intentionally not counted in running chunk length
            curr_chunk, curr_chunk_len = fwd_overlap, 0
            back_overlap, back_overlap_len = [], 0  # Start gathering
            fwd_overlap, fwd_overlap_len = None, 0  # Stop gathering

        if fwd_overlap is not None:
            fwd_overlap.append(s)
            fwd_overlap_len += split_len

        if back_overlap is not None:
            back_overlap.append(s)
            back_overlap_len += split_len

        curr_chunk.append(s)
        curr_chunk_len += split_len

    # Done with the splits; use the final back_overlap, if any
    if back_overlap:
        prev_chunk.extend(back_overlap)

    if at_least_one_chunk:
        yield separator.join(prev_chunk)
    else:
        # Degenerate case where no splits found & chunk size too large; just one big chunk
        yield text


def text_splitter(text: str,
        chunk_size: int,
        chunk_overlap: int=0,
        separator: str='\n\n',
        len_func=len
    ) -> list[str]:
    '''
    Split string into a sequence of chunks in a "fuzzy" manner. Will generally not split the text into sequences
    of chunk_size length, but will instead preserve some overlap on either side of the given separators.
    This results in slightly larger chunks than the chunk_size number by itself suggests.

    >>> from ogbujipt.text_helper import text_splitter
    >>> chunks = text_split_fuzzy('she sells seashells by the seashore', chunk_size=50, separator=' ')

    Args:
        text (str): (Multiline) String to be split into chunks

        chunk_size (int): Number of characters to include per chunk

        chunk_overlap (int, optional): Number of characters to overlap at the edges of chunks

        seperator (str, optional): String that already splits "text" into sections

        distance_function (callable, optional): Function to measure length, len() by default

    Returns:
        chunks (List[str]): List of chunks of the text provided
    '''
    warnings.warn('text_splitter() is deprecated. Use text_split_fuzzy() instead.')

    assert separator, 'Separator must be non-empty'
    
    if ((not isinstance(text, str))
        or (not isinstance(separator, str))):
        raise ValueError(f'text and separator must be strings.\n'
                         f'Got {text.__class__} for text and {separator.__class__} for separator')
    
    if chunk_overlap == 0:
        msg = 'chunk_overlap must be a positive integer. For no overlap, use text_split() instead'
        raise ValueError(msg)

    if ((not isinstance(chunk_size, int))
        or (not isinstance(chunk_overlap, int))
        or (chunk_size <= 0)
        or (chunk_overlap < 0)
        or (chunk_size < chunk_overlap)):
        raise ValueError(f'chunk_size must be a positive integer, '
                         f'chunk_overlap must be a non-negative integer, and'
                         f'chunk_size must be greater than chunk_overlap.\n'
                         f'Got {chunk_size} chunk_size and {chunk_overlap} chunk_overlap.')
    
    # Split up the text by the separator
    # FIXME: Need a step for escaping regex
    sep_pat = re.compile(separator)
    fine_split = re.split(sep_pat, text)
    separator_len = len_func(separator)

    if len(fine_split) <= 1:
        warnings.warn(f'No splits detected. Perhaps a problem with separator? ({repr(separator)})?')

    # Combine the small pieces into medium size chunks
    # chunks will accumulate processed text chunks as we go along
    # curr_chunk will be a list of subchunks comprising the main, current chunk
    # back_overlap will be appended, once ready, to the end of the previous chunk (if any)
    # fwd_overlap will be prepended, once ready, to the start of the next chunk
    chunks = []
    curr_chunk, curr_chunk_len = [], 0
    back_overlap, back_overlap_len = None, 0  # None signals not yet gathering
    fwd_overlap, fwd_overlap_len = None, 0

    for s in fine_split:
        if not s: continue  # noqa E701
        split_len = len_func(s) + separator_len
        # Check for full back_overlap (if relevant, i.e. back_overlap isn't None)
        if back_overlap is not None and (back_overlap_len + split_len > chunk_overlap):  # noqa: F821
            chunks[-1].extend(back_overlap)
            back_overlap, back_overlap_len = None, 0

        # Will adding this split take us into overlap room?
        if curr_chunk_len + split_len > (chunk_size - chunk_overlap):
            fwd_overlap, fwd_overlap_len = [], 0  # Start gathering

        # Will adding this split take us over chunk size?
        if curr_chunk_len + split_len > chunk_size:
            # If so, complete current chunk & start a new one

            # fwd_overlap should be non-None at this point, so check empty
            if not fwd_overlap and curr_chunk:
                # If empty, look back to make sure there is some overlap
                fwd_overlap.append(curr_chunk[-1])

            chunks.append(curr_chunk)
            # fwd_overlap intentionally not counted in running chunk length
            curr_chunk, curr_chunk_len = fwd_overlap, 0
            back_overlap, back_overlap_len = [], 0  # Start gathering
            fwd_overlap, fwd_overlap_len = None, 0  # Stop gathering

        if fwd_overlap is not None:
            fwd_overlap.append(s)
            fwd_overlap_len += split_len

        if back_overlap is not None:
            back_overlap.append(s)
            back_overlap_len += split_len

        curr_chunk.append(s)
        curr_chunk_len += split_len

    # Done with the splits; use the final back_overlap, if any
    if back_overlap:
        chunks[-1].extend(back_overlap)

    # Concatenate all the split parts of all the chunks
    chunks = [separator.join(c) for c in chunks]

    # Handle degenerate case where no splits found & chunk size too large
    # Just becomes one big chunk
    if not chunks:
        chunks = [text]

    # chunks.append(separator.join(curr_chunk))
    return chunks
