'''
Quick demo, sending an Alpaca-compatible LLm some bad XML & asking it
to make corrections.

Needs access to an OpenAI-like service. Default assumption is that you
have a self-hosted framework such as llama-cpp-python or text-generation-webui
running. Say it's at my-llm-host:8000, you can do:

python demo/alpaca_simple_fix_xml.py --host=http://my-llm-host --port=8000
'''

import click
from langchain import OpenAI

from ogbujipt.config import openai_emulation
from ogbujipt.model_style.alvic import make_prompt, sub_style


# Command line arguments defined in decorators
@click.command()
@click.option('--host', default='http://127.0.0.1', help='OpenAI API host')
@click.option('--port', default='8000', help='OpenAI API port')
def main(host, port):
    # Set up API connector
    openai_emulation(host=host, port=port)
    llm = OpenAI(temperature=0.1)

    BAD_XML_CODE = '''\
<earth>
<country><b>Russia</country></b>
<capital>Moscow</capital>
</Earth>'''

    prompt = make_prompt(
        'Correct the following XML to make it well-formed',
        inputs=BAD_XML_CODE,
        sub=sub_style.ALPACA_INSTRUCT
        )
    # print(prompt)

    response = llm(prompt)
    print(response)


# CLI entry point
if __name__ == '__main__':
    main()