# SPDX-FileCopyrightText: 2023-present Uche Ogbuji <uche@ogbuji.net>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.text_helper

'''
Routines to help with text processing
'''
import re
import warnings
from typing import List


def text_splitter(text, chunk_size, chunk_overlap, separator='\n\n',
                  len_func=len) -> List[str]:
    '''
    Split text into a set of chunks

    Much like langchain's CharTextSplitter.py

    >>> from ogbujipt.text_helper import text_splitter
    >>> from PyPDF2 import PdfReader
    >>> pdf_reader = PdfReader('monopoly-board-game-manual.pdf')
    >>> text = ''.join((page.extract_text() for page in pdf_reader.pages))
    >>> chunks = text_splitter(text, chunk_size=500, separator='\n')

    Args:
        text (str): (Multiline) String to be split into chunks

        chunk_size (int): Number of characters to include per chunk

        chunk_overlap (int): Number of characters to overlap at the edges of chunks

        seperator (str, optional): String that already splits "text" into sections

        distance_function (callable, optional): Function to measure length, len() by default

    Returns:
        chunks (List[str]): List of chunks of the text provided
    '''
    assert separator  # FIXME: Clean up error handling
    if chunk_overlap is None:
        # 10% overlap by default
        chunk_overlap = chunk_size // 10

    if not (isinstance(chunk_size, int) and chunk_overlap > 0):
        raise ValueError(f'''\
chunk_size must be an integer greater than 0. Got {chunk_overlap}''')

    if not (isinstance(chunk_overlap, int)
            and chunk_overlap > 0 and chunk_overlap < chunk_size):
        raise ValueError(f'''\
chunk_overlap must be an integer greater than 0 and smaller than chunk_size. \
Got {chunk_overlap}''')
    # Split up the text by the separator
    # FIXME: Need a step for escaping regex
    sep_pat = re.compile(separator)
    fine_split = re.split(sep_pat, text)
    separator_len = len_func(separator)

    if len(fine_split) <= 1:
        warnings.warn(
            f'No splits detected. Problem with separator ({repr(separator)})?')

    # Combine the small pieces into medium size chunks to send to LLM
    # Initialize accumulators; chunks will be the target list of the chunks so far
    # curr_chunk will be a list of parts comprising the main, current chunk
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
