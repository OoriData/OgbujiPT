# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.model_styles.async_helper

'''
Coroutines to make it a little easier to multitask LLM access
using Python asyncio
'''
import sys
import asyncio

async def console_progress_indicator(pause=0.5, file=sys.stderr):
    '''
    Simple progress indicator for the console. Just prints dots.

    pause - seconds between each dot printed to console, default half a sec

    file - file for dots output, default STDERR
    '''
    while True:
        print('.', end='', flush=True, file=file)
        await asyncio.sleep(pause)
