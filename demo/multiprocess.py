'''
Advanced demo showing quick chat with an LLM, but with 3 simultaneous requests,
and also a separate, progress indicator dislay while the LLM instances are generating.
Key is taking advantage of Python's asyncio, and also multiprocess, which requires some finesse,
to work even when the LLM framework in use doesn't suport asyncio.
Luckily `ogbujipt.async_helper` comes in handy.

```sh
python demo/alpaca_multitask_fix_xml.py --host=http://my-llm-host --port=8000
```

Also allows you to use the actual OpenAI ChatGPT service, by specifying --openai
'''
import asyncio

# import openai

import click

from ogbujipt import oapi_first_choice_text
from ogbujipt import config
from ogbujipt.async_helper import (
    schedule_callable,
    openai_api_surrogate,
    console_progress_indicator,
    save_openai_api_params)
from ogbujipt.prompting.basic import format
from ogbujipt.prompting.model_style import ALPACA_DELIMITERS


class llm_request:
    '''
    Encapsulates each LLM service request via OpenAI API (even for self-hosted LLM)
    '''
    tasks = {}

    def __init__(self, topic, llmtemp, **model_params):
        '''
        topic - a particular topic about which we'll ask the LLM
        model_params - mapping of custom parameters for model behavior, e.g.:
            max_tokens: limit number of generated tokens (default 16)
            top_p: AKA nucleus sampling; can increase generated text diversity
            frequency_penalty: Favor more or less frequent tokens
            presence_penalty: Prefer new, previously unused tokens
            More info: https://platform.openai.com/docs/api-reference/completions
        '''
        self.topic = topic
        self.llmtemp = llmtemp
        self.model_params = model_params

    def wrap(self):
        prompt = format(f'Tell me a funny joke about {self.topic}', delimiters=ALPACA_DELIMITERS)

        # Pattern of passing in the callable iself, then the params—required for multiprocess execution
        self.task = asyncio.create_task(
            schedule_callable(openai_api_surrogate, prompt, temperature=self.llmtemp,
                              **self.model_params, **save_openai_api_params()))
        llm_request.tasks[self.task] = self
        return self.task


async def async_main(topics, llmtemp):
    # Pro tip: When creating tasks with asyncio.create_task be mindful to not
    # accidentally lose references to tasks, lest they get garbage collected,
    # which sows chaos. In some cases asyncio.TaskGroup (new in Python 3.11)
    # is a better alternative, but we can't use them in this case because
    # they wait for all tasks to complete whereas we're done once only
    # the LLM generation task is complete
    indicator_task = asyncio.create_task(console_progress_indicator())
    # Notice the pattern of passing in the callable iself, then the params
    # You can't just do, say llm(prompt) because that will actually
    # call the function & block on the LLM request
    llm_requests = [llm_request(t, llmtemp, max_tokens=1024) for t in topics]
    llm_tasks = [req.wrap() for req in llm_requests]
    # Need to gather to make sure all LLm tasks are completed
    gathered_llm_tasks = asyncio.gather(*llm_tasks)
    done, _ = await asyncio.wait((indicator_task, gathered_llm_tasks), return_when=asyncio.FIRST_COMPLETED)

    # Completed task will from gather() of llm_tasks; results in original task arg order
    results = zip(llm_requests, next(iter(done)).result())
    for req, resp in results:
        print(f'Result re {req.topic}')
        # resp is an instance of openai.openai_object.OpenAIObject, with lots of useful info
        print('\nFull response data from LLM:\n', resp)
        # Just the response text
        response_text = oapi_first_choice_text(resp)
        print('\nResponse text from LLM:\n\n', response_text)


# Command line arguments defined in click decorators
@click.command()
@click.option('--host', default='http://127.0.0.1', help='OpenAI API host')
@click.option('--port', default='8000', help='OpenAI API port')
@click.option('--llmtemp', default='0.9', type=float, help='LLM temperature')
@click.option('--openai', is_flag=True, default=False, type=bool,
              help='Use live OpenAI API. If you use this option, you must have '
              '"OPENAI_API_KEY" defined in your environmnt')
@click.option('--model', default='', type=str, 
              help='OpenAI model to use (see https://platform.openai.com/docs/models)')
def main(host, port, llmtemp, openai, model):
    # Use OpenAI API if specified, otherwise emulate with supplied host, etc.
    if openai:
        model = model or 'text-davinci-003'
        config.openai_live(model=model, debug=True)
    else:
        # Generally not really useful except in conjunction with --openai
        model = model or config.HOST_DEFAULT
        config.openai_emulation(host=host, port=port, model=model, debug=True)

    topics = ['wild animals', 'vehicles', 'space aliens']

    asyncio.run(async_main(topics, llmtemp))


if __name__ == '__main__':
    # CLI entry point. Also protects against re-execution of main() after process fork
    # viz https://docs.python.org/3/library/multiprocessing.html#multiprocessing-safe-main-import
    main()
