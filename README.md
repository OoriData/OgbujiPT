# OgbujiPT

Toolkit for using self-hosted large language models, through langchain & other means

Includes demos with chat-your-documents and AGI/AutoGPT/privateGPT-style capabilities.

The main, tested front-end is langchain, but this can be used in a very shallow way to interact with LLMs (see, e.g. the `alpaca_simple_fix_xml.py` demo where really only 3 lines are langchain-specific).

Tested back ends are [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) or [text-generation-webui](https://github.com/oobabooga/text-generation-webui) (AKA Oobabooga or Ooba). In our own practice we use boh of these with Nvidia GPU.

<!--
Not yet in PyPI
[![PyPI - Version](https://img.shields.io/pypi/v/ogbujipt.svg)](https://pypi.org/project/ogbujipt)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ogbujipt.svg)](https://pypi.org/project/ogbujipt)
-->
## Quick links

- [Getting started]([#installation](getting-started))
- [License](#license)

-----

## Getting started

```console
pip install --upgrade .
```

<!--
Not yet in PyPI
```console
pip install ogbujipt
```

-->

## A bit more explanation

Many self-hosted AI large language models are now astonishingly good, even running on consumer-grade hardware, which provides an alternative for those of us who would rather not be sending all our data out over the network to the likes of ChatGPT & Bard. OgbujiPT provides a toolkit for using and experimenting with LLMs via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) or [text-generation-webui](https://github.com/oobabooga/text-generation-webui) (AKA Oobabooga or Ooba), a popular tool for self-hosting such models. OgbujiPT can invoke these to complete prompted tasks on self-hosted LLMs.

* [Quick setup for llama-cpp-python](https://github.com/uogbuji/OgbujiPT/wiki/Quick-setup-for-llama-cpp-python-backend)
* [Quick setup for Ooba](https://github.com/uogbuji/OgbujiPT/wiki/Quick-setup-for-text-generation-webui-(Ooba)-backend)

Right now OgbujiPT requires a bit of Python development on the user's part, but more general capabilities are coming.

## Bias to good software engineering

I've seen many projects talking stabs at something like this one, but they really just seem to be stabs, usually by folks interested in LLM who admit they don't have strong coding backgrounds. This not only leads to a lumpy patchwork of forks and variations, as people try to figure out the narrow, gnarly paths that cater to their own needs, but also hampers maintainability just at a time when everything seems to be changing drastically every few days.

I have a strong Python and software engineering background, and I'm looking to apply that in this project, to hopefully create something more easily speclailized for other needs, built-upon, maintained and contributed to.

This project is packaged using [hatch](https://hatch.pypa.io/), a modern Python packaging tool. I plan to write tests as I go along, and to incorporate continuous integration. Admit I may be slow to find the cycles for all that, but at least the intent and architecture is there from the beginning.

## Model styles

A central concept of OgbujiPT is model styles. There are [numerous Open LLMs out there](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), and each one tends to have some specialization, including in how prompt it. Here we define model styles to help encapsulate these differences, and make it easier to quickly launch experiments, adapt to and adopt other models.

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

Enh?! Yo mama! 😝 My surname is Ogbuji, so it's a bit of a pun: Ooba + GPT by Ogbuji = OgbujiPT, ya feel me?
