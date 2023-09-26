'''
Advanced demo showing quick chat with LLMs, with 3 simultaneous requests.
It can be configured to chat with 3 separate LLMs, or 3 instances of the same LLM.
Includes a progress indicator dislay while the LLM instances are generating.
Takes advantage of Python's asyncio, and also multiprocess, which requires some finesse,
mostly handled by OgbujiPT. Works even when the LLM host in use doesn't suport asyncio.

```sh
python demo/multiprocess.py --apibase=http://my-llm-host:8000
```

Also allows you to use the actual OpenAI ChatGPT service, by specifying --openai
'''
import asyncio
from pathlib import Path

import click

from ogbujipt import config
from ogbujipt import word_loom
from ogbujipt import oapi_chat_first_choice_message
from ogbujipt.llm_wrapper import openai_api, openai_chat_api, prompt_to_chat
from ogbujipt.async_helper import (
    schedule_callable,
    openai_api_surrogate,
    console_progress_indicator,
    save_openai_api_params)
from ogbujipt.prompting.basic import format
from ogbujipt.prompting.model_style import ALPACA_DELIMITERS


def file_path_here():
    '''Cross-platform Python trick to get the path to this very file'''
    from inspect import getsourcefile
    from os.path import abspath

    return abspath(getsourcefile(lambda: 0))


# Load language material, such as prompt templates, from an external file—word loom format
HERE = Path(file_path_here()).parent
with open(HERE / Path('language.toml'), mode='rb') as fp:
    TEXTS = word_loom.load(fp)


# XXX: Probably doesn't require another class layer, now that we have llm_wrapper
class llm_request:
    '''
    Encapsulates each LLM service request via OpenAI API (including for self-hosted LLM)
    '''
    tasks = {}
    request_tpl = TEXTS['test_prompt_joke']

    def __init__(self, llm_wrapped, topic, llmtemp, **model_params):
        '''
        topic - a particular topic about which we'll ask the LLM
        model_params - mapping of custom parameters for model behavior, e.g.:
            llm_wrapped: LLM wrapper to use (e.g. ogbujipt.llm_wrapper.openai_api)
            max_tokens: limit number of generated tokens (default 16)
            top_p: AKA nucleus sampling; can increase generated text diversity
            frequency_penalty: Favor more or less frequent tokens
            presence_penalty: Prefer new, previously unused tokens
            More info: https://platform.openai.com/docs/api-reference/completions
        '''
        self.topic = topic
        self.llmtemp = llmtemp
        self.model_params = model_params
        self.llm_wrapped = llm_wrapped

    def wrap(self):
        prompt = format(self.request_tpl.format(topic=self.topic), delimiters=ALPACA_DELIMITERS)

        # Pattern: schedule the callable, passing in params separately—for non-blocking multiprocess execution
        self.task = asyncio.create_task(
            schedule_callable(self.llm_wrapped, prompt_to_chat(prompt), temperature=self.llmtemp,
                              **self.model_params))
        llm_request.tasks[self.task] = self
        return self.task


async def async_main(requests_info):
    '''
    Main entry point for asyncio
    model_topics - list of (llm_wrapped, topic, temperature) tuples
    llmtemp - LLM temperature; hard-coded across all (see https://beta.openai.com/docs/api-reference/completions/create)
    '''
    # Pro tip: When creating asyncio tasks be mindful to not lose references to tasks,
    # lest they get garbage collected, which sows chaos. asyncio.TaskGroup (new in Python 3.11)
    # is often a better alternative, but waits for all tasks to complete whereas we're done once
    # the LLM generation tasks are complete
    indicator_task = asyncio.create_task(console_progress_indicator())
    llm_requests = [llm_request(w, topic, temp, max_tokens=1024) for (w, topic, temp) in requests_info]
    llm_tasks = [req.wrap() for req in llm_requests]
    # Need to gather to make sure all LLM tasks are completed
    gathered_llm_tasks = asyncio.gather(*llm_tasks)
    done, _ = await asyncio.wait((indicator_task, gathered_llm_tasks), return_when=asyncio.FIRST_COMPLETED)

    # Completed task will from gather() of llm_tasks; results in original task arg order
    results = zip(llm_requests, next(iter(done)).result())
    for req, resp in results:
        print(f'Result re {req.topic}')
        # resp is an instance of openai.openai_object.OpenAIObject, with lots of useful info
        print('\nFull response data from LLM:\n', resp)
        # Just the response text
        response_text = oapi_chat_first_choice_message(resp)
        print('\nResponse text from LLM:\n\n', response_text)


# Command line arguments defined in click decorators
@click.command()
@click.option('--apibase', default='http://127.0.0.1:8000', help='OpenAI API base URL')
@click.option('--llmtemp', default='0.9', type=float, help='LLM temperature')
@click.option('--openai', is_flag=True, default=False, type=bool,
              help='Use live OpenAI API. If you use this option, you must have '
              '"OPENAI_API_KEY" defined in your environmnt')
@click.option('--model', default='', type=str, 
              help='OpenAI model to use (see https://platform.openai.com/docs/models)')
def main(apibase, llmtemp, openai, model):
    # Use OpenAI API if specified, otherwise emulate with supplied host, etc.
    if openai:
        model = model or 'gpt-3.5-turbo'
        oapi = openai_chat_api(model=model)
    else:
        model = model or config.HOST_DEFAULT
        oapi = openai_chat_api(model=model, api_base=apibase)

    # Separate models or params—e.g. temp—for each LLM request is left as an exercise 😊
    requests_info = [(oapi, t, llmtemp) for t in ('wild animals', 'vehicles', 'space aliens')]

    asyncio.run(async_main(requests_info))


if __name__ == '__main__':
    # CLI entry point. Also protects against re-execution of main() after process fork
    # viz https://docs.python.org/3/library/multiprocessing.html#multiprocessing-safe-main-import
    main()
