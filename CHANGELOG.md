# Changelog

Notable changes to OgbujiPT. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
## [Unreleased]
-->
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
