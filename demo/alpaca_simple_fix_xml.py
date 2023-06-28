'''
Quick demo, sending an Alpaca-compatible LLm some bad XML & asking it
to make corrections.

Needs access to an OpenAI-like service. Default assumption is that you
have a self-hosted framework such as llama-cpp-python or text-generation-webui
running. Say it's at my-llm-host:8000, you can do:

python demo/alpaca_simple_fix_xml.py --host=http://my-llm-host --port=8000
'''

import click
import os
# from langchain import OpenAI
import openai
from ogbujipt.config import openai_live
# from ogbujipt.model_style.alvic import make_prompt, sub_style
from ogbujipt.prompting.basic import context_build, pdelim


# Command line arguments defined in decorators
@click.command()
@click.option('--host', default='http://127.0.0.1', help='OpenAI API host')
@click.option('--port', default='8000', help='OpenAI API port')
def main(host, port):
    # Set up API connector
    # openai_emulation(host=host, port=port)
    openai_live()

    # llm = OpenAI(temperature=0.1)
    # print(os.environ["API_KEY"])
    BAD_XML_CODE = '''\
<earth>
<country><b>Russia</country></b>
<capital>Moscow</capital>
</Earth>'''

    vicuna_delimiters = {
        pdelim.PREQUERY: '### USER',
        pdelim.POSTQUERY: '### ASSISTANT',
    }

    prompt = context_build(
        f'Correct the following XML to make it well-formed\n\n{BAD_XML_CODE}',
        preamble='You are a helpful assistant, who answers questions briefly, in 1st grade language',
        delimiters=vicuna_delimiters)
    print(prompt)

    openai.api_key = 'BOGUS'
    #API key is stopping openAI
    # print(openai.api_key)

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=1,
        max_tokens=60,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=1
        )
    #response is a json and needs to be parsed for the text
    print(response)


# CLI entry point
if __name__ == '__main__':
    main()
