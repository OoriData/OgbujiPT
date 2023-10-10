# OgbujiPT

Toolkit for using self-hosted large language models (LLMs), but also with support for full-service such as ChatGPT.

Includes demos with RAG ("chat your documents") and AGI/AutoGPT/privateGPT-style capabilities, via streamlit, Discord, command line, etc.

There are some helper functions for common LLM tasks, such as those provided by projects such as langchain, but not meant to be as extensive. The OgbujiPT approach emphasizes simplicity and transparency.

Tested back ends are [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), [text-generation-webui](https://github.com/oobabooga/text-generation-webui) (AKA Oobabooga or Ooba) and in-memory hosted LLaMA-class (and more) models via [ctransformers](https://github.com/marella/ctransformers). In our own practice we apply these with Nvidia and Apple M1/M2 GPU enabled.

We also test with OpenAI's full service GPT (3, 3.5, and 4) APIs, and apply these in our practice.

<table><tr>
  <td><a href="https://oori.dev/"><img src="https://www.oori.dev/assets/branding/ooriLogo_FullColor.png" width="64" /></a></td>
  <td>OgbujiPT is primarily developed by the crew at <a href="https://oori.dev/">Oori Data</a>. We offer software engineering services around LLM applications.</td>
</tr></table>

[![PyPI - Version](https://img.shields.io/pypi/v/ogbujipt.svg)](https://pypi.org/project/ogbujipt)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ogbujipt.svg)](https://pypi.org/project/ogbujipt)

## Quick links

- [Getting started](#getting-started)
- [License](#license)

-----

## Getting started

```console
pip install ogbujipt
```

### Just show me some code, dammit!

from ogbujipt.llm_wrapper import ctrans_wrapper
model = AutoModelForCausalLM.from_pretrained(
        '/Users/uche/.local/share/models/TheBloke_LlongOrca-13B-16K-GGUF',
        model_file='llongorca-13b-16k.Q5_K_M.gguf',
        model_type="llama",
        gpu_layers=50)
oapi = ctrans_wrapper(model=model)
print(oapi('The quick brown fox'))

```py
from ogbujipt.llm_wrapper import openai_api
from ogbujipt import oapi_first_choice_text
from ogbujipt.prompting import format, ALPACA_INSTRUCT_DELIMITERS

llm_api = openai_api(api_base='http://localhost:8000')  # Update for your LLM API host
# Change the delimiters to a prompting style that suits the LLM you're using
prompt = format('Write a short birthday greeting for my star employee',
                delimiters=ALPACA_INSTRUCT_DELIMITERS)

# You can set model params as needed
response = llm_api(prompt=prompt, temperature=0.1, max_tokens=100)
# Extract just the response text, but the entire structure is available
print(oapi_first_choice_text(response))
```

The [Nous-Hermes 13B](https://huggingface.co/TheBloke/Nous-Hermes-13B-GGML) LLM offered the following response:

> Dear [Employee's Name],
> I hope this message finds you well on your special day! I wanted to take a moment to wish you a very happy birthday and express how much your contributions have meant to our team. Your dedication, hard work, and exceptional talent have been an inspiration to us all.
> On this occasion, I want you to know that you are appreciated and valued beyond measure. May your day be filled with joy and laughter.

Here's an example using a model loaded in memory using ctransformers, a LLaMMa-based model (so ultimately via llama.cpp).

```py
from ctransformers import AutoModelForCausalLM

from ogbujipt.llm_wrapper import ctransformer as ctrans_wrapper

model = AutoModelForCausalLM.from_pretrained('TheBloke_LlongOrca-13B-16K-GGUF',
        model_file='llongorca-13b-16k.Q5_K_M.gguf', model_type="llama", gpu_layers=50)
llm = ctrans_wrapper(model=model)

print(llm(prompt='Write a short birthday greeting for my star employee', max_new_tokens=100))
```

For more examples see the [demo directory](https://github.com/uogbuji/OgbujiPT/tree/main/demo)

## A bit more explanation

Many self-hosted AI large language models are now astonishingly good, even running on consumer-grade hardware, which provides an alternative for those of us who would rather not be sending all our data out over the network to the likes of ChatGPT & Bard. OgbujiPT provides a toolkit for using and experimenting with LLMs as loaded into memory via or via OpenAI API-compatible network servers such as:

* [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
* [text-generation-webui](https://github.com/oobabooga/text-generation-webui) (AKA Oobabooga or Ooba)

OgbujiPT can invoke these to complete prompted tasks on self-hosted LLMs. It can also be used for building front ends to ChatGPT and Bard, if these are suitable for you.

* [Quick setup for llama-cpp-python](https://github.com/uogbuji/OgbujiPT/wiki/Quick-setup-for-llama-cpp-python-backend)
* [Quick setup for Ooba](https://github.com/uogbuji/OgbujiPT/wiki/Quick-setup-for-text-generation-webui-(Ooba)-backend)

Right now OgbujiPT requires a bit of Python development on the user's part, but more general capabilities are coming.

## Bias to sound software engineering

I've seen many projects taking stabs at something like this one, but they really just seem to be stabs, usually by folks interested in LLM who admit they don't have strong coding backgrounds. This not only leads to a lumpy patchwork of forks and variations, as people try to figure out the narrow, gnarly paths that cater to their own needs, but also hampers maintainability just at a time when everything seems to be changing drastically every few days.

I have a strong Python and software engineering background, and I'm looking to apply that in this project, to hopefully create something more easily speclailized for other needs, built-upon, maintained and contributed to.

This project is packaged using [hatch](https://hatch.pypa.io/), a modern Python packaging tool. I plan to write tests as I go along, and to incorporate continuous integration. Admit I may be slow to find the cycles for all that, but at least the intent and architecture is there from the beginning.

## Prompting patterns

Different LLMs have different conventions you want to use in order to get high
quality responses. If you've looked into [self-hosted LLMs](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) you might have heard
of the likes of alpaca, vicuña or even airoboros. OgbujiPT includes some shallow
tools in order to help construct prompts according to the particular conventions
that would be best for your choice of LLM. This makes it easier to quickly launch
experiments, adapt to and adopt other models.

# Contributions

For reasons I'm still investigating (some of the more recent developments and issues in Python packaging are [quite esoteric](https://chriswarrick.com/blog/2023/01/15/how-to-improve-python-packaging/)), some of the hatch tools such as `hatch run` are problematic. I suspect they might not like the way I rename directories during build, but I won't be compromising on that. So, for example, to run tests, just stick to:

```shell
pytest test
```

More [notes for contributors in the wiki](https://github.com/uogbuji/OgbujiPT/wiki/Notes-for-contributors).

# License

Apache 2. For tha culture!

# Credits

Some initial ideas & code were borrowed from these projects, but with heavy refactoring:

* [ChobPT/oobaboogas-webui-langchain_agent](https://github.com/ChobPT/oobaboogas-webui-langchain_agent)
* [wafflecomposite/langchain-ask-pdf-local](https://github.com/wafflecomposite/langchain-ask-pdf-local)

# FAQ

- [What's unique about this toolkit?](#whats-unique-about-this-toolkit)
- [Does this support GPU for locally-hosted models](#does-this-support-gpu-for-locally-hosted-models)
- [What's with the crazy name?](#whats-with-the-crazy-name)

## What's unique about this toolkit?

I mentioned the bias to software engineering, but what does this mean?

* Emphasis on modularity, but seeking as much consistency as possible
* Support for multitasking
* Finding ways to apply automated testing

## Does this support GPU for locally-hosted models

Yes, but you have to make sure you set up your back end LLm server (llama.cpp or text-generation-webui) with GPU, and properly configure the model you load into it. If you can use the webui to query your model and get GPU usage, that will also apply here in OgbujiPT.

Many install guides I've found for Mac, Linux and Windows touch on enabling GPU, but the ecosystem is still in its early days, and helpful resouces can feel scattered.

* [Quick setup for llama-cpp-python](https://github.com/uogbuji/OgbujiPT/wiki/Quick-setup-for-llama-cpp-python-backend)
* [Quick setup for Ooba](https://github.com/uogbuji/OgbujiPT/wiki/Quick-setup-for-text-generation-webui-(Ooba)-backend)

## What's with the crazy name?

Enh?! Yo mama! 😝 My surname is Ogbuji, so it's a bit of a pun.
This is the notorious OGPT, ya feel me?
