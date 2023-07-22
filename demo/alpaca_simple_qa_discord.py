'''
Advanced demo of a Discord chatbot with an LLM back end

Demonstrates async processing via ogbujipt.async_helper & Discord API integration.
Users can make an LLM request by @mentioning the bot by its user ID

Note: This is a simple demo, which doesn't do any client-side job management,
so for example if a request is sent, and a second comes in before it has completed,
the LLM back end is relied on to cope.

Prerequisites: python-dotenv discord.py

You also need to make sure Python has root SSL certificates installed
On MacOS this is via double-clicking `Install Certificates.command`

You also need to have a file, just named `.env`, in the same directory,
with contents such as:

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

For hints on how to modify this to use OpenAI's actual services,
see demo/alpaca_simple_fix_xml.py
'''

import os
import asyncio

import discord
from dotenv import load_dotenv

from ogbujipt.config import openai_emulation
from ogbujipt.async_helper import schedule_callable, openai_api_surrogate, save_openai_api_params
from ogbujipt import oapi_first_choice_text
from ogbujipt.prompting.basic import format
from ogbujipt.prompting.model_style import ALPACA_DELIMITERS

# Enable all standard intents, plus message content
# The bot app you set up on Discord will require this intent (Bot tab)
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


async def send_llm_msg(msg):
    '''
    Schedule the LLM request
    '''
    prompt = format(msg, delimiters=ALPACA_DELIMITERS)
    print(prompt, '\n')

    # See demo/alpaca_multitask_fix_xml.py for some important warnings here
    llm_task = asyncio.create_task(
        schedule_callable(openai_api_surrogate, prompt, temperature=llmtemp, max_tokens=512,
                          **save_openai_api_params()))

    tasks = [llm_task]
    done, _ = await asyncio.wait(
        tasks, return_when=asyncio.FIRST_COMPLETED)

    response = next(iter(done)).result()

    # Response is a json-like object; extract the text
    print('\nFull response data from LLM:\n', response)

    # Response is a json-like object; 
    # just get back the text of the response
    response_text = oapi_first_choice_text(response)
    print('\nResponse text from LLM:\n', response_text)

    return response_text


@client.event
async def on_message(message):
    # Ignore the bot's own messages & respond only to @mentions
    # The client.user.id check creens out @everyone & @here pings
    # FIXME: Better content check—what if the bot's id is a common word?
    if message.author == client.user \
            or not client.user.mentioned_in(message) \
            or str(client.user.id) not in message.content:
        return

    # Send throbber placeholder message to discord:
    return_msg = await message.channel.send('<a:oori_throbber:1119445227732742265>')

    # Assumes a single mention, for simplicity. If there are multiple,
    # All but the first will just be bundled over to the LLM
    mention_str = f'<@{client.user.id}>'
    clean_msg = message.content.partition(mention_str)
    clean_msg = clean_msg[0] + clean_msg[2]

    response = await send_llm_msg(clean_msg)

    await return_msg.edit(content=response[:2000])  # Discord messages cap at 2k characters


@client.event
async def on_ready():
    print(f"Bot is ready. Connected to {len(client.guilds)} guild(s).")


def main():
    # A real app would probably use a discord.py cog w/ these as data members
    global llm, llmtemp

    load_dotenv()  # From .env file
    DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
    assert DISCORD_TOKEN
    LLM_HOST = os.getenv('LLM_HOST', 'http://localhost')
    LLM_PORT = os.getenv('LLM_PORT', 8000)

    # Set up API connector & update temperature from environment
    llm = openai_emulation(host=LLM_HOST, port=LLM_PORT)
    llmtemp = float(os.getenv('LLM_TEMP', '0.9'))

    # launch Discord client event loop
    client.run(DISCORD_TOKEN)


if __name__ == '__main__':
    # Entry point protects against multiple launching of the overall program
    # when a child process imports this 
    # viz https://docs.python.org/3/library/multiprocessing.html#multiprocessing-safe-main-import
    main()
