# Changelog

Notable changes to OgbujiPT. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
## [Unreleased]
-->

## [Unreleased]

### Added

- GGUF support in download-model.py
- Switch to a class-based wrapper for LLM endpoints/handlers - #39
- Add support for in-memory LLM loading via ctransformers
- Added postgreSGL vector support to embedding_helper.py as new Class `PGvectorConnection`
  - PGvectorConnection is a wrapper around [asyncpg](https://github.com/MagicStack/asyncpg), and is primarily just capable of excecuting raw SQL queries right now.
  - There are a few common SQL queries included in the class for using PGv as a vector database, but they are not yet fully tested.
  - Added a demonstration notebook which uses `PGvectorConnection` to do similarity search

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

- `ogbujipt.__version__`
- chat_web_selects.py demo
- `ogbujipt.async_helper.save_openai_api_params()`

### Fixed

- chat_pdf_streamlit_ui.py demo
- OpenAI API reentrancy & async_helper.py

### Changed

- Renamed demo alpaca_simple_fix_xml.py → simple_fix_xml.py
- Renamed demo alpaca_multitask_fix_xml.py → multiprocess.py
- Renamed `ogbujipt.oapi_choice1_text()` → `ogbujipt.oapi_first_choice_text()`
- Renamed `ogbujipt.async_helper.schedule_llm_call()` → `ogbujipt.async_helper.schedule_callable()`

## [0.1.1] - 20230711

### Added

- GitHub CI workflow
- Orca model style
- Convenience function ogbujipt.oapi_choice1_text()
- Additional conveniences in ogbujipt.prompting.model_style

### Fixed

- Linter cleanup

### Changed

- Qdrant embeddings interface
- Renamed ogbujipt.prompting.context_build() → ogbujipt.prompting.format()

## [0.1.0]

- Initial standalone release candidate
