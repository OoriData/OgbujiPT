'''
More advanced demo using an LLM to repair data (XML), like
alpaca_simple_fix_xml.py
but rnning a separate, progress indicator task in the background
while the LLm works, using asyncio. This should work even
if the LLM framework we're using doesn't suport asyncio,
thanks to ogbujipt.async_helper 

You need access to an OpenAI-like service. Default assumption is that you
have a self-hosted framework such as llama-cpp-python or text-generation-webui
running. Say it's at my-llm-host:8000, you can do:

python demos/alpaca_multitask_fix_xml.py --host=http://my-llm-host --port=8000
'''

import asyncio

import click
from langchain import OpenAI

from ogbujipt.config import openai_emulation
from ogbujipt.async_helper import schedule_llm_call
from ogbujipt.model_style.alpaca import prep_instru_inputs, ALPACA_PROMPT_TMPL

DOTS_SPACING = 0.5  # Number of seconds between each dot printed to console


# Could probably use something like tqdm.asyncio, if we wanted to be fancy
async def indicate_progress(pause):
    '''
    Simple progress indicator for the console. Just prints dots.
    '''
    while True:
        print('.', end='', flush=True)
        await asyncio.sleep(pause)


async def async_main(llm):
    '''
    Schedule one task to do a long-running/blocking LLM request, and another
    to run a progress indicator in the background
    '''
    BAD_XML_CODE = '''\
    <earth>    
    <country><b>Russia</country></b>
    <capital>Moscow</capital>
    </Earth>'''

    instru_inputs = prep_instru_inputs(
        'Correct the following XML to make it well-formed',
        inputs=BAD_XML_CODE
        )

    prompt = ALPACA_PROMPT_TMPL.format(instru_inputs=instru_inputs)
    # print(prompt)

    # Pro tip: Always be mindful when creating tasks with asyncio.create_task
    # That you don't accidentally lose references to them, because they
    # Can get garbage collected, which sows chaos
    # In some cases asyncio.TaskGroup (new in Python 3.11) are a better alternative,
    # But we can't use them in this case because they wait for all tasks to complete
    # Whereas we want to be done once only the LLM call is complete
    indicator_task = asyncio.create_task(indicate_progress(DOTS_SPACING))
    llm_task = asyncio.create_task(schedule_llm_call(llm, prompt))
    tasks = [indicator_task, llm_task]
    done, _ = await asyncio.wait(
        tasks, return_when=asyncio.FIRST_COMPLETED
        )

    print('\nResponse from LLM: ', next(iter(done)).result())


# Command line arguments defined in decorators
@click.command()
@click.option('--host', default='http://127.0.0.1', help='OpenAI API host')
@click.option('--port', default='8000', help='OpenAI API port')
def main(host, port):
    # Set up API connector
    openai_emulation(host=host, port=port)
    llm = OpenAI(temperature=0.1)
    asyncio.run(async_main(llm))


if __name__ == '__main__':
    # CLI entry point
    # Also protects against multiple launching of the overall program
    # when a child process imports this
    # viz https://docs.python.org/3/library/multiprocessing.html#multiprocessing-safe-main-import
    main()
