# Changelog

Notable changes to  Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
## [Unreleased]

-->

## [0.9.2] - 20240625

### Added

- `joiner` param to `text_helper.text_split()` for better control of regex separator handling
- query filter mix-in, `embedding.pgvector.match_oneof()`, for use with `meta_filter` argument to `DB.search`
- `llm_wrapper.response_type` to differentiate tool calling vs regular LLM responses

### Changed

- Index word loom items by their literal default language text, as well
- Cleaned up PGVector query-building logic

### Fixed

- `llm_wrapper.llm_response` objects to handle tool calls
- failure of some matching scenarios in `embedding.pgvector.match_exact()`

### Removed

- Previously deprecated `first_choice_text` & `first_choice_message` methods

## [0.9.1] - 20240604

### Fixed

- Demos & testing around `text_helper.text_split_fuzzy()`

## [0.9.0] - 20240604

### Added

- `text_helper.text_split_fuzzy()` as basically a generator version of `text_helper.text_splitter()`, and deprecate the latter
- `text_helper.text_split()` which brooks no overlap
- `embedding.pgvector_data` module for vector database embedding of data fields using PGVector. Document fields are now treated as simple text, with specialized metadata generalized in the new `metadata` field
- query filter mix-in, `embedding.pgvector.match_exact()`, for filtering vector search by metadata fields via new `meta_filter` argument to DB.search. This restores some of the tag matching search functionality that's been removed.
- `clone()` method on `wordloom.language_item` class (formerly `text_item`)
- `meta` and `preserve_key` args on `wordloom.language_item` (formerly `text_item`) initializer, to preserve TOML table items and top-level TOML key for each language item, respectively, in object properties

### Changed

- Deprecation of `text_helper.text_splitter()`
- `tags` field of table encapsulated in `embedding.pgvector_data` now modified into a general, JSON `meta` field
- Word Loom now uses `_` as the TOML table key for text content. The former `text` is deprecated
- Word Loom now uses `_m` as the TOML table key for text substitution markers. The former `markers` is deprecated
- Word Loom now reserves TOML table keys beginning with `_`, and passes on all other keys to the new `meta` property
- `wordloom.text_item` class now renamed `wordloom.language_item`
- `lang` arg on `wordloom.language_item` (formerly `text_item`) initializer renamed `deflang`

### Removed

- `embedding.pgvector_data_doc` (now just `embedding.pgvector_data`)
- `conjunctive` option for tags searching now removed, in favor of a query filter mix-in approach

## [0.8.0] - 20240325

### Added

- `llm_wrapper.llama_cpp_http_chat` & `llm_wrapper.llama_cpp_http`; llama.cpp low-level HTTP API support
- `llm_wrapper.llama_response` class with flexible handling across API specs
- `window` init param for for `embedding.pgvector.MessageDB`, to limit message storage per history key
- `threshold` param on `MessageDB.search()`

### Changed

- Deprecated `first_choice_text` & `first_choice_message` methods in favor of `first_choice_text` attributes on response objects
- Clarify test suite setup docs

## [0.7.1] - 20240229

### Added

- MessageDB.get_messages() options: `since` (for retrieving messages aftter a timestamp) and `limit` (for limiting the number of messages returned, selecting the most recent)

### Changed

- PGVector users now manage their own connection pool by default
- Better modularization of embeddings test cases; using `conftest.py` more
- `pgvector_message.py` PG table timstamp column no longer a primary key

### Fixed

- Backward threshold check for embedding.pgvector_data_doc.DataDB

## [0.7.0] - 20240110

### Added

- Command line options for `demo/chat_web_selects.py`
- Helper for folks installing on Apple Silicon: `constraints-apple-silicon.txt`
- Function calling demo
- `embedding.pgvector_message.insert_many()`

### Changed

- Improved use of PGVector helper SQL query parameters
- PGVector helper `search(query_tags=[..])` now uses contains operator (filters by existence in tag sets), not the same as where tags are OR
- PGVector helper `search` can now be set to work conjunctively or disjunctively
- PGVector helper `query` now has threshold arg based on degree of similarity. `limit` default now unlimited. Use SQL query args for query_embedding.
- `embedding.pgvector` split into a couple of modules.
- Separated data-style PGVector DBs from doc-style. tags is no longer the final param for PGVector docs helper methods & some params renamed.
- PGVector helper method results now as `attr_dict`
- PGVector helper now uses connection pooling & is more multiprocess safe
- `embedding.pgvector_chat` renamed to `embedding.pgvector_message`
- DB MIGRATION REQUIRED - `embedding.pgvector_message` table schema

### Fixed

- `insert_many` PGVector helper method; semantics & performance
- `demo/chat_web_selects.py` & `demo/chat_pdf_streamlit_ui.py` (formerly non-functional)
- Tests & CI for PGVector helper

## [0.6.1] - 20231114

### Changed

- Use PG timestamp rather than serial for chat logs

### Removed

- test/test_text_w_apostrophe.ipynb (incorporated into test/embedding/test_pgvector.py)

## [0.6.0] - 20231113

### Added

- Support for efficient multi-queries (`executemany`): `insert_many` vs `insert`
- Chatlog-specific PGVector helper (`PGvectorHelper` specialized into `DocDB` & `MessageDB`)
- PG Vector DB instance launch fo ruse in test suite & GitHub actions
- Updated model styles and prompt formatting, particularly for improved closed-context patterns & per-context metadata (e.g. for Airboros)

### Changed

- Model introspection moved to llm_wrapper classes: `hosted_model` & `available_models`
- Move OAI API response structure handling helpers to be static methods of the llm_wrapper classes
- Clarified demo names
- Support upstream python-openai > 1.0

### Fixed

- README sample code
- Demos
- Test cases
- Use of string formatting intead of SQL query parameters
- Registration of vector type
- pgvector test case

## [0.5.1] - 20231010

### Fixed

- `embedding_helper.py` logic

## [0.5.0] - 20230919

### Added

- Support for GGUF in download-model.py
- Support for in-memory LLM loading via ctransformers
- PostgreSGL vector support to embedding_helper.py, new class `PGvectorConnection`
  - PGvectorConnection is a wrapper around [asyncpg](https://github.com/MagicStack/asyncpg), and is primarily just capable of excecuting raw SQL queries right now.
  - There are a few common SQL queries included in the class for using PGv as a vector database, but they are not yet fully tested.
  - Added a demonstration notebook which uses `PGvectorConnection` to do similarity search
- `oapi_first_choice_content` function

### Changed

- Switch to a class-based wrapper for LLM endpoints/handlers - #39

### Fixed

- Model style tweaks

## [0.4.0] - 20230728

### Added

- Initial implementation of [Word Loom](https://github.com/OoriData/OgbujiPT/wiki/Word-Loom:-A-format-for-managing-language-for-AI-LLMs-(including-prompts)) (see, for example, demo/multiprocess.py)
- More tests to improve coverage
- Qadrant collection reset method (embedding_helper.py)

### Fixed

- Input parameter controls in embedding_helper.py

### Changed

- test suite structure

## [0.3.0] - 20230723

### Added

- `__version__`
- chat_web_selects.py demo
- `async_helper.save_openai_api_params()`

### Fixed

- chat_pdf_streamlit_ui.py demo
- OpenAI API reentrancy & async_helper.py

### Changed

- Renamed demo alpaca_simple_fix_xml.py → simple_fix_xml.py
- Renamed demo alpaca_multitask_fix_xml.py → multiprocess.py
- Renamed `oapi_choice1_text()` → `oapi_first_choice_text()`
- Renamed `async_helper.schedule_llm_call()` → `async_helper.schedule_callable()`

## [0.1.1] - 20230711

### Added

- GitHub CI workflow
- Orca model style
- Convenience function oapi_choice1_text()
- Additional conveniences in prompting.model_style

### Fixed

- Linter cleanup

### Changed

- Qdrant embeddings interface
- Renamed prompting.context_build() → prompting.format()

## [0.1.0]

- Initial standalone release candidate
