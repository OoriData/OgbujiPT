# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
#
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.embedding
'''
Vector DBs are useful when you have a lot of context to use with LLMs,
e.g. a large document or collection of docs. One common pattern is to create
vector indices on this text. Given an LLM prompt, the vector DB is first queried
to find the most relevant "top k" sections of text based on the prompt,
which are added as context in the ultimate LLM invocation.

For sample code see demo/chat_pdf_streamlit_ui.py

You need an LLM to turn the text into vectors for such indexing, and these
vectors are called the embeddings. You can usually create useful embeddings
with a less powerful (and more efficient) LLM.

This package provides utilities to set up a vector DB, and use it to index
chunks of text using a provided LLM model to create the embeddings.

Other common use-cases for vector DBs in LLM applications:

* Long-term LLM Memory for chat: index the entire chat history and retrieve
the most relevant and recent N messages based on the user's new message,
to give the LLM a chance to make its responses coherent and on-topic

* Cache previous LLM interactions, saving resources by retrieving previous
responses to similar questions without having to use the most powerful LLM
'''
