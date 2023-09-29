# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.llm_wrapper
'''
Wrapper for LLMs, including via OpenAI API and locally hosted models

Side note: a lot of OpenAI tutorials suggest that you embed your
OpenAI private key into the code, which is a horrible, no-good idea

Extra reminder: If you set up your environment via .env file, make sure
it's in .gitignore or equivalent so it never gets accidentally committed!
'''

import os
import asyncio
import concurrent.futures
from functools import partial

try:
    import openai as openai_api_global
except ImportError:
    openai_api_global = None

from ogbujipt import config


# FIXME: Should be an ABC
class llm_wrapper:
    '''
    Base-level wrapper for LLMs
    '''
    def __init__(self, model=None, **kwargs):
        '''
        Args:
            model (str): Name of the model being wrapped

            kwargs (dict, optional): Extra parameters for the API, the model, etc.
        '''
        self.model = model
        self.parameters = kwargs


# Extracted from https://github.com/openai/openai-python/blob/main/openai/__init__.py
OPENAI_GLOBALS = ['api_key', 'api_key_path', 'api_base', 'organization', 'api_type', 'api_version',
                 'proxy', 'app_info', 'debug', 'log']


class openai_api(llm_wrapper):
    '''
    Wrapper for LLM hosted via OpenAI-compatible API (including OpenAI proper).
    Designed for models that provide simple completions from prompt.
    For chat-style models (including OpenAI's gpt-3.5-turbo & gpt-4), use openai_chat_api
    '''
    def __init__(self, model=None, api_base=None, api_key=None, **kwargs):
        '''
        If using OpenAI proper, you can pass in an API key, otherwise environment variable
        OPENAI_API_KEY will be checked

        This class is designed such that you can have multiple instances with different LLMs wrapped,
        perhaps one is OpenAI proper, another is a locally hosted model, etc. (a good way to save costs)
        This also opens up reentrancy concerns, especially if you're using multi-threading or multi-processing.

        If multi-processing, this class should bundle its settings correctly across the fork,
        and when you call methods in the child process, these settings should be swapped into global context correctly

        Args:
            model (str, optional): Name of the model being wrapped. Useful for using
            OpenAI proper, or any endpoint that allows you to select a model

            api_base (str, optional): Base URL of the API endpoint

            api_key (str, optional): OpenAI API key to use for authentication

            debug (bool, optional): Debug flag

        Returns:
            openai_api (openai): Prepared OpenAI API

        Args:
            model (str): Name of the model being wrapped

            kwargs (dict, optional): Extra parameters for the API, the model, etc.
        '''
        if openai_api_global is None:
            raise ImportError('openai module not available; Perhaps try again after: `pip install openai`')
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY', config.OPENAI_KEY_DUMMY)

        self.api_key = api_key
        self.model = model
        self.parameters = kwargs
        self.api_base = api_base
        self.full_api_base = api_base + kwargs.get('api_version', '/v1')
        self._claim_global_context()

    # Nature of openai library requires that we babysit their globals
    def _claim_global_context(self):
        '''
        Set global context of the OpenAI API according to this class's settings
        '''
        for k in OPENAI_GLOBALS:
            if hasattr(self.parameters, k):
                setattr(openai_api_global, k, getattr(self.parameters, k))
        openai_api_global.model = self.model
        openai_api_global.api_key = self.api_key
        openai_api_global.api_base = self.full_api_base

    def __call__(self, prompt, api_func=None, **kwargs):
        '''
        Invoke the LLM with a completion request

        Args:
            prompt (str): Prompt to send to the LLM

            kwargs (dict, optional): Extra parameters to pass to the model via API

        Returns:
            dict: JSON response from the LLM
        '''
        api_func = api_func or openai_api_global.Completion.create
        # Ensure the right context, e.g. after a fork or when using multiple LLM wrappers
        self._claim_global_context()
        merged_kwargs = {**self.parameters, **kwargs}
        return api_func(model=self.model, prompt=prompt, **merged_kwargs)

    def wrap_for_multiproc(self, prompt, **kwargs):
        '''
        Wrap the LLM invocation in an asyncio task

        Returns:
            asyncio.Task: Task for the LLM invocation
        '''
        merged_kwargs = {**self.parameters, **kwargs}
        # print(f'wrap_for_multiproc: merged_kwargs={merged_kwargs}')
        return asyncio.create_task(
            schedule_callable(self, prompt, **merged_kwargs))


class openai_chat_api(openai_api):
    '''
    Wrapper for a chat-style LLM hosted via OpenAI-compatible API (including OpenAI proper).
    Supports local chat-style models as well as OpenAI's gpt-3.5-turbo & gpt-4
    '''
    def __call__(self, messages, api_func=None, **kwargs):
        '''
        Invoke the LLM with a completion request

        Args:
            messages (list): Series of messages representing the chat history
            e.f. messages=[{"role": "user", "content": "Hello world"}])

            kwargs (dict, optional): Extra parameters to pass to the model via API

        Returns:
            dict: JSON response from the LLM
        '''
        api_func = api_func or openai_api_global.ChatCompletion.create
        # Ensure the right context, e.g. after a fork or when using multiple LLM wrappers
        self._claim_global_context()
        return api_func(model=self.model, messages=messages, **self.parameters, **kwargs)


def prompt_to_chat(prompt):
    '''
    Convert a prompt string to a chat-style message list

    Args:
        prompt (str): Prompt to convert

    Returns:
        list: single item with just the given prompt as user message for chat history
        e.g. messages=[{"role": "user", "content": "Hello world"}])
    '''
    # return [{'role': 'user', 'content': m} for m in prompt.split('\n')]
    return [{'role': 'user', 'content': prompt}]


async def schedule_callable(callable, *args, **kwargs):
    '''
    Schedule long-running/blocking function call in a separate process,
    wrapped to work well in an asyncio event loop

    Basically hides away a bunch of the multiprocessing webbing

    e.g. `llm_task = asyncio.create_task(schedule_callable(llm, prompt))`

    Can then use asyncio.wait(), asyncio.gather(), etc. with `llm_task`

    Args:
        callable (callable): Callable to be scheduled

    Returns:
        response: Response object
    '''
    # print(f'schedule_callable: kwargs={kwargs}')
    # Link up the current async event loop for multiprocess execution
    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ProcessPoolExecutor()
    # Need to partial execute to get in any kwargs for the target callable
    prepped_callable = partial(callable, **kwargs)
    # Spawn a separate process for the LLM call
    response = await loop.run_in_executor(executor, prepped_callable, *args)
    return response
