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
import openai as openai_api  # pip install openai
from dotenv import load_dotenv
from ogbujipt.config import openai_live, openai_emulation
from ogbujipt.prompting.basic import context_build
from ogbujipt.prompting.model_style import VICUNA_DELIMITERS


# Command line arguments defined in decorators
@click.command()
@click.option('--host', default='http://127.0.0.1', help='OpenAI API host')
@click.option('--port', default='8000', help='OpenAI API port')
@click.option('--llmtemp', default='0.1', type=float, help='LLM temperature')
@click.option('--openai', is_flag=True, default=False, type=bool,
              help='Use live OpenAI API. If you use this option, you must have ' +
              '"OPENAI_API_KEY" defined in your environmnt')
@click.option('--model', default='', type=str, help='OpenAI model to use (see https://platform.openai.com/docs/models)')
def main(host, port, llmtemp, openai, model):
    # Use OpenAI API if specified, otherwise use host as user defines
    if openai:
        openai_api.api_key = os.getenv('OPENAI_API_KEY')
        openai_live(debug=True)
        model = model or 'text-ada-001'
        openai_model = {'model': model}
    else:
        # Emulate OpenAI API with "host" and "port" for LLM call
        openai_api.api_key = 'BOGUS'
        openai_emulation(host=host, port=port)
        model = model or 'LOCAL'
        openai_model = {}

    BAD_XML_CODE = '''\
<earth>
<country><b>Russia</country></b>
<capital>Moscow</capital>
</Earth>'''

    prompt = context_build(
        f'Correct the following XML to make it well-formed\n\n{BAD_XML_CODE}',
        preamble='You are a helpful assistant, who answers questions briefly, in 1st grade language',
        delimiters=VICUNA_DELIMITERS)
    print(prompt, '\n')
    print(openai_api.api_key)

    response = openai_api.Completion.create(
        host=host,
        port=port,
        model="bogus",  # Model (Required)
        # model="text-davinci-003",  # Model (Required)
        prompt=prompt,  # Prompt (Required)
        temperature=llmtemp,  # Temp (Default 1)
        max_tokens=60,  # Max Token length of generated text (Default no max)
        top_p=1,  # Also known as nucleus sampling, 
                  # This can help to increase the diversity of the generated
                  # text (Default 1)
        frequency_penalty=0,    # influences the model to favor more or less
                                # frequent tokens (Default 0)
        presence_penalty=1  # influences the model to use new tokens it has
                            # not yet used (Default 0)
        )
    
    # Response is a json and this is how you extract the text
    print(response["choices"][0]["text"])


# CLI entry point
if __name__ == '__main__':
    main()
