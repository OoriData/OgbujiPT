'''
Advanced demo of a Discord chatbot with an LLM back end

Demonstrates:
* Async processing via ogbujipt.async_helper
* Discord API integration
* Client-side serialization of requests to the server. An important
consideration until more sever-side LLM hosting frameworks reliablly
support multiprocessing

You need access to an OpenAI-like service. Default assumption is that you
have a self-hosted framework such as llama-cpp-python or text-generation-webui
running. Say it's at my-llm-host:8000, you can do:

Prerequisites: python-dotenv discord.py

You also need to have a file, just named `.env`, in the same directory,
with contents such as:

You also need ot make sure Python has root SSL certificates installed
On Mac this is via double-clicking `Install Certificates.command`

```env
DISCORD_TOKEN={your-bot-token}
LLM_HOST=http://my-llm-host
LLM_PORT=8000
LLM_TEMP=0.5
```

Then to launch the bot:

```shell
python demo/alpaca_simple_qa_discord.py
```
'''

import os
import asyncio

import discord
from dotenv import load_dotenv

from langchain import OpenAI

from ogbujipt.config import openai_emulation
from ogbujipt.async_helper import schedule_llm_call
from ogbujipt.model_style.alpaca import prep_instru_inputs, ALPACA_PROMPT_TMPL

# Enable all standard intents, plus message content
# The bot app you set up on Discord will require this intent (Bot tab)
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


class llm_manager:
    def __init__(self, llm):
        self.llm = llm


async def send_llm_msg(msg):
    '''
    Schedule the LLM request
    '''
    prompt = ALPACA_PROMPT_TMPL.format(
        instru_inputs=prep_instru_inputs(msg))
    print(prompt)

    llm = llm_man.llm

    # See demo/alpaca_multitask_fix_xml.py for some important warnings here
    llm_task = asyncio.create_task(schedule_llm_call(llm, prompt))
    tasks = [llm_task]
    done, _ = await asyncio.wait(
        tasks, return_when=asyncio.FIRST_COMPLETED
        )

    response = next(iter(done)).result()
    print('\nResponse from LLM: ', response)
    return response


@client.event
async def on_message(message):
    # Ignore the bot's own messages & respond only to @mentions
    # The client.user.id check creens out @everyone & @here pings
    # FIXME: Better content check—what if the bot's id is a common word?
    if message.author == client.user \
            or not client.user.mentioned_in(message) \
            or str(client.user.id) not in message.content:
        return

    # Assumes a single mention, for simplicity. If there are multiple,
    # All but the first will just be bundled over to the LLM
    mention_str = f'<@{client.user.id}>'
    clean_msg = message.content.partition(mention_str)
    clean_msg = clean_msg[0] + clean_msg[2]

    response = await send_llm_msg(clean_msg)

    await message.channel.send(response)


def main():
    global llm_man  # Ick! Ideally should be better scope/context controlled

    load_dotenv()  # From .env file
    DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
    LLM_HOST = os.getenv('LLM_HOST')
    LLM_PORT = os.getenv('LLM_PORT')
    LLM_TEMP = os.getenv('LLM_TEMP')

    # Set up API connector
    openai_emulation(host=LLM_HOST, port=LLM_PORT)
    llm = OpenAI(temperature=LLM_TEMP)

    llm_man = llm_manager(llm)
    client.run(DISCORD_TOKEN)


if __name__ == '__main__':
    # Entry point protects against multiple launching of the overall program
    # when a child process imports this
    # viz https://docs.python.org/3/library/multiprocessing.html#multiprocessing-safe-main-import
    main()
