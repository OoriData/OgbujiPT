# OgbujiPT

Simple, BabyAGI-like toolkit using locally hosted LLMs, via [Oobabooga](https://github.com/oobabooga/text-generation-webui)

<!--
Not yet in PyPI
[![PyPI - Version](https://img.shields.io/pypi/v/ogbujipt.svg)](https://pypi.org/project/ogbujipt)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ogbujipt.svg)](https://pypi.org/project/ogbujipt)
-->
## Quick links

- [Getting started]([#installation](getting-started))
- [License](#license)

-----

## A bit more explanation

What do I mean by BabyAGI-like? BabyAGI is one of the projects using LLM state of the art to experiment with what one could cheekily call Artificial General Intelligence. Most such projects call out to LLMs on the cloud, such as OpenAI's ChatGPT, but some of us would rather not be sending all our data out there, especially when many self-hosted LLMs are now astonishingly good, even running on consumer-grade hardware.

This project assumes you can set up Ooba, a popular tool for self-hosting such models, and [have it set up with an OpenAI-like API (which you can configure right from Ooba)](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/openai). It then calls out to that self-hosted LLM API to complete prompt tasks you give it. Docker is a handy option for Ooba, and how I host it for myself.

## Bias to good software engineering

I've seen many projects talking stabs at something like this one, but they really just seem to be stabs, usually by folks interested in LLM who admit they don't have strong coding backgrounds. This not only leads to a lumpy patchwork of forks and variations, as people try to figure out the narrow, gnarly paths that cater to their own needs, but also hampers maintainability just at a time when everything seems to be changing drastically every few days.

I have a strong Python and software engineering background, and I'm looking to apply that in this project, to hopefully create something more easily speclailized fo rotehr needs, built-upon, maintained and contributed to.

For example, this project if packaged using [hatch](https://hatch.pypa.io/), a very modern Python packaging tool. I plan to write tests as I go along, and to incorporate continuous integration. I admit I may be slow to find the cycles for all that, but at least the intent and architecture is there from the beginning.

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

# License

Apache 2. For tha culture!

# Credits

Some initial ideas & code were borrowed from these projects, but with heavy refactoring:

* [ChobPT/oobaboogas-webui-langchain_agent](https://github.com/ChobPT/oobaboogas-webui-langchain_agent)
* [sebaxzero/LangChain_PDFChat_Oobabooga](https://github.com/sebaxzero/LangChain_PDFChat_Oobabooga)

# FAQ

- [Does it support GPU for locally-hosted models](#does-it-support-gpu-for-locally-hosted-models)
- [What's with the crazy name?](#whats-with-the-crazy-name)

## Does this support GPU for locally-hosted models

Yes, but you have to make sure you set up your system and text-generation-webui install with GPU, and properly configure the model you load into it (via the WebUI or launch arguments). I may try to write up some sort of helper docs for that, at some point, because I had to find tidbits from all over the place, and experiment madly, in order to get it to work myself. But if you can use the webui to query your model and get GPU use, that will also apply here in OgbujiPT.

## What's with the crazy name?

Eh?! Yo mama! 😝 My surname is Ogbuji, so it's a bit of a pun: Ooba + GPT by Ogbuji = OgbujiPT, ya feel me?
