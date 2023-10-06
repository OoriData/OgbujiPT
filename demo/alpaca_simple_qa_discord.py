# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt/demo/alpaca_simple_qa_discord.py
'''
Advanced demo of a Discord chatbot with an LLM back end

Demonstrates async processing via ogbujipt.async_helper & Discord API integration.
Users can make an LLM request by @mentioning the bot by its user ID

Note: This is a simple demo, which doesn't do any client-side job management,
so for example if a request is sent, and a second comes in before it has completed,
the LLM back end is relied on to cope.

Additional prerequisites: discord.py

Also need to make sure Python has root SSL certificates installed
On MacOS this is via double-clicking `Install Certificates.command`

Requires the following environment variables:

```env
DISCORD_TOKEN={your-bot-token}
LLM_BASE=http://my-llm-host:8000
LLM_TEMP=0.5
```

For some discussion of setting up the environment: https://github.com/OoriData/OgbujiPT/discussions/36

Then to launch the bot:

```shell
python demo/alpaca_simple_qa_discord.py
```

You can then @mention the bot in a Discord channel where it's been added & chat with it

For hints on how to modify this to use OpenAI's actual services,
see demo/alpaca_simple_fix_xml.py
'''

import os

import discord

from ogbujipt import config, oapi_first_choice_text
from ogbujipt.llm_wrapper import openai_api
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
    oapi.parameters
    response = await oapi.wrap_for_multiproc(prompt, max_tokens=512)
    print(response)

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
    return_msg = await message.channel.send('<a:oori_throbber:1142173241499197520>')

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
    # A real app would probably use a discord.py cog w/ this as data member
    global oapi

    DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
    assert DISCORD_TOKEN
    llm_base = os.getenv('LLM_BASE', 'http://localhost:8000')
    llmtemp = float(os.getenv('LLM_TEMP', '0.9'))

    # Set up API connector; OpenAI API emulation with supplied API base, fixed temp, from the environment
    oapi = openai_api(model=config.HOST_DEFAULT, api_base=llm_base, temperature=llmtemp)

    # launch Discord client event loop
    client.run(DISCORD_TOKEN)


if __name__ == '__main__':
    # Entry point protects against multiple launching of the overall program
    # when a child process imports this 
    # viz https://docs.python.org/3/library/multiprocessing.html#multiprocessing-safe-main-import
    main()
