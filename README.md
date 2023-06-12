# OgbujiPT

AGI/AutoGPT-style toolkit using self-hosted LLMs (via langchain & self-hosted models—GPU capable)

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

# License

Apache 2. For tha culture!

# Credits

Some initial ideas & code were borrowed from these projects, but with heavy refactoring:

* [ChobPT/oobaboogas-webui-langchain_agent](https://github.com/ChobPT/oobaboogas-webui-langchain_agent)
* [wafflecomposite/langchain-ask-pdf-local](https://github.com/wafflecomposite/langchain-ask-pdf-local)

# FAQ

- [Does this support GPU for locally-hosted models](#does-this-support-gpu-for-locally-hosted-models)
- [What's with the crazy name?](#whats-with-the-crazy-name)

## Does this support GPU for locally-hosted models

Yes, but you have to make sure you set up your system and text-generation-webui install with GPU, and properly configure the model you load into it (via the WebUI or launch arguments). I may try to write up some sort of helper docs for that, at some point, because I had to find tidbits from all over the place, and experiment madly, in order to get it to work myself. But if you can use the webui to query your model and get GPU usage, that will also apply here in OgbujiPT.

## What's with the crazy name?

Enh?! Yo mama! 😝 My surname is Ogbuji, so it's a bit of a pun: Ooba + GPT by Ogbuji = OgbujiPT, ya feel me?
