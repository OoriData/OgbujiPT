OgbujiPT Contributor Guide

# Quick Reference

## Why We Use `uv pip install -U .`

This project uses a source layout where `pylib/` becomes `ogbujipt/` during package building. This remapping only happens during wheel building, not in development environments.

**Why not use hatch environments?**
- Hatch's path remapping (`tool.hatch.build.sources`) only applies during wheel building
- Hatch's dev-mode uses editable installs which can't apply the source remapping
- Setting `dev-mode=false` means no install happens at all

**Solution:** We use proper package installation (`uv pip install -U .`) instead of editable/dev-mode installs. This ensures the source remapping is applied correctly and your development environment matches the built package.

## Daily Development

```bash
# Install in current virtualenv
uv pip install -U .

# Install with all test dependencies (includes PostgreSQL/PGVector, Qdrant, etc.)
uv pip install -U ".[testall]"

# Run tests (uses in-memory stores, skips PostgreSQL integration tests)
pytest test/ -v

# Run specific test file
pytest test/test_llm_wrapper.py -v

# Run vector store tests (fast, no PostgreSQL needed)
pytest test/store/ -v

# Run integration tests (requires PostgreSQL with PGVector)
pytest test/ -v -m integration

# Skip integration tests (default)
pytest test/ -v -m "not integration"

# Run linting
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Run tests with coverage
pytest test/ --cov=ogbujipt --cov-report=html
```

## Making Changes

```bash
# After editing any Python files in pylib/
uv pip install -U .

# After editing resources/
uv pip install -U .

# After editing tests only (no reinstall needed)
pytest test/ -v
```

## Useful Commands

```bash
# See package structure after install
python -c "import ogbujipt, os; print(os.path.dirname(ogbujipt.__file__))"
ls -la $(python -c "import ogbujipt, os; print(os.path.dirname(ogbujipt.__file__))")

# Check what files are in the installed package
pip show -f ogbujipt

# Check installed version
python -c "import ogbujipt; print(ogbujipt.__version__)"

# Compare source version
cat pylib/__about__.py

# Uninstall completely
pip uninstall ogbujipt -y

# Clean build artifacts
rm -rf build/ dist/ *.egg-info
rm -rf .pytest_cache .ruff_cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
```

## Testing Package Build Locally

```bash
# Build locally
python -m build
python -m build -w  # For some reason needs to need both, in this order. Probably an issue in how we're using hatch

# Test the built wheel
pip install dist/ogbujipt-0.X.Y-py3-none-any.whl --force-reinstall

# Check package contents
unzip -l dist/ogbujipt-0.X.Y-py3-none-any.whl
```

# Project Structure

```
OgbujiPT/
├── pylib/              # Source code (becomes 'ogbujipt' package)
│   ├── __init__.py
│   ├── __about__.py    # Version info
│   ├── llm/            # LLM wrapper implementations
│   ├── retrieval/      # Retrieval (dense, sparse, hybrid)
│   ├── store/          # Vector stores (PostgreSQL/PGVector, Qdrant)
│   ├── memory/         # Memory management
│   ├── text/           # Text processing utilities
│   ├── ingestion/      # Document ingestion
│   ├── observability/  # Logging and observability
│   └── mcp/            # Model Context Protocol
├── resources/          # Bundled resources
│   └── ogbujipt/
├── test/               # Tests
│   ├── test_config.py
│   ├── test_llm_wrapper.py
│   ├── test_ogbujipt.py
│   ├── test_text_splitter.py
│   └── store/          # Store-specific tests
├── demo/               # Example applications
├── pyproject.toml      # Project config
└── README.md
```

When installed, becomes:

```
site-packages/
└── ogbujipt/
    ├── __init__.py
    ├── __about__.py
    ├── llm/
    ├── retrieval/
    ├── store/
    ├── memory/
    ├── text/
    ├── ingestion/
    ├── observability/
    ├── mcp/
    └── resources/
        └── ogbujipt/
```

## Key Files

- `pylib/__about__.py` - Version number (update for releases)
- `pyproject.toml` - Dependencies, metadata, build config
- `README.md` - Main documentation
- `CHANGELOG.md` - Release notes
- `LICENSE` - Apache 2.0 license

# Publishing a Release

Before creating a release:

- [ ] Update version in `pylib/__about__.py`
- [ ] Update CHANGELOG.md
- [ ] Run tests locally: `pytest test/ -v -m "not integration"` (or with integration tests if you have PostgreSQL set up)
- [ ] Run linting: `ruff check .`
- [ ] Verify package builds: `python -m build`
- [ ] Commit and push all changes
<!-- 
- [ ] Create git tag: `git tag v0.X.Y`
- [ ] Push tag: `git push origin v0.X.Y`
 -->
- [ ] [Create GitHub release](https://github.com/OoriData/OgbujiPT/releases/new) (triggers publish workflow)
- [ ] Verify package update on PyPI: https://pypi.org/project/ogbujipt/

## Testing the Package

After publishing, test the installation:

```bash
# Create a fresh virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from PyPI
pip install ogbujipt

# Test import
python -c "import ogbujipt; print(ogbujipt.__version__)"

# Test basic functionality
python -c "
from ogbujipt.llm_wrapper import openai_chat_api, prompt_to_chat

llm_api = openai_chat_api(base_url='http://localhost:8000')
prompt = 'Hello, world!'
resp = llm_api.call(prompt_to_chat(prompt), temperature=0.1, max_tokens=50)
print(resp.first_choice_text)
"
```

# Initial Project Setup

Historical, and to inform maintenance. GitHub Actions & PyPI publishing.

## GitHub Actions Setup

The repository includes two workflows:

### 1. CI Workflow (`.github/workflows/main.yml`)

Runs automatically on every push and pull request. It:
- Tests on Python 3.12 and 3.13
- Sets up PostgreSQL with PGVector for integration tests
- Runs ruff linting
- Runs pytest test suite

### 2. Publish Workflow (`.github/workflows/publish.yml`)

Runs when you create a new GitHub release. It builds and publishes to PyPI using trusted publishing (OIDC).

## PyPI Trusted Publishing Setup

###  PyPI Setup

- Login your [PyPI](https://pypi.org) account
- For new package:
    - Go to: https://pypi.org/manage/account/publishing/
    - Click "Add a new pending publisher"
    - Fill in:
    - **PyPI Project Name**: `OgbujiPT` (must match `name` in `pyproject.toml`, with case)
    - **Owner**: `OoriData`
    - **Repository name**: `OgbujiPT`
    - **Workflow name**: `publish.yml`
    - **Environment name**: `pypi` (PyPI's recommended name)
- If the package already exists on PyPI:
    - Go to the project page: https://pypi.org/manage/project/ogbujipt/settings/publishing/
    - Add the publisher configuration as above

### GitHub Setup
- Go to: https://github.com/OoriData/OgbujiPT/settings/environments
- Click "New environment"
- Name: `pypi`
- Click "Configure environment"
- (Optional) Add protection rules:
    - Required reviewers: Add yourself to require manual approval before publishing
    - Wait timer: Add a delay (e.g., 5 minutes) before publishing
- Click "Save protection rules"

### Note on using the environment name

Using an environment name (`pypi`) adds an extra layer of protection, with rules such as required reviewers (manual approval before publishing), wait timers (delay before publishing) and branch restrictions. Without an environment stipulation the workflow runs automatically when a release is created.

## First Time Publishing

Option on the very first release to PyPI: may want to do a manual publish to ensure everything is set up correctly:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# For some reason, the wheel only seems to work if you build first without then with `-w`
python -m build -w

# Basic build check
twine check dist/*

# Extra checking
VERSION=0.10.0 pip install --force-reinstall -U dist/ogbujipt-$VERSION-py3-none-any.whl
python -c "import ogbujipt; print(ogbujipt.__version__)"

# Upload to Test PyPI first (optional but recommended)
twine upload --repository testpypi dist/*
# Username: __token__
# Password: your-test-pypi-token

# If test looks good, upload to real PyPI
twine upload dist/*
# Username: __token__
# Password: your-pypi-token
```

After the first manual upload, you can use trusted publishing for all future releases.

## Troubleshooting

### "Project name 'OgbujiPT' is not valid"
- Check that the name in `pyproject.toml` matches exactly
- Names are case-sensitive and must match what you registered on PyPI

### "Invalid or non-existent authentication information"
- For trusted publishing: Double-check the repository name, owner, and workflow name
- For token auth: Make sure the token is saved as `PYPI_API_TOKEN` in GitHub secrets

### Workflow fails with "Resource not accessible by integration"
- Make sure the workflow has `id-token: write` permission
- Check that the repository settings allow GitHub Actions

### Package version already exists
- You can't overwrite versions on PyPI
- Increment the version in `pylib/__about__.py` and create a new release

## Testing Strategies

### In-Memory Testing (Default)

By default, vector store tests use **in-memory implementations** requiring zero setup:

```bash
# Fast tests, no external dependencies
pytest test/store/ -v          # ~0.5 seconds

# All tests (skips integration by default)
pytest test/ -v
```

**Benefits:**
- ✅ Zero setup - works everywhere
- ✅ Lightning fast execution
- ✅ Perfect for CI/CD
- ✅ Ideal for rapid iteration

The in-memory stores (`RAMDataDB`, `RAMMessageDB`) are full-featured and also available to users for prototyping. See `demo/ram-store/` for examples.

### Integration Testing (Optional PostgreSQL)

Integration tests verify behavior against real PostgreSQL with pgvector. These are **skipped by default** but useful for:
- Validating database-specific features
- Testing migration paths
- Performance benchmarking

**Setup PostgreSQL for Integration Tests:**

```bash
# Option 1: Docker (recommended)
docker run -d -p 5432:5432 \
  -e POSTGRES_USER=mock_user \
  -e POSTGRES_PASSWORD=mock_password \
  -e POSTGRES_DB=mock_db \
  pgvector/pgvector:pg17

# Option 2: Set environment variables for existing PostgreSQL
export PG_DB_HOST='localhost'
export PG_DB_NAME='test_db'
export PG_DB_USER='user'
export PG_DB_PASSWORD='pass'
export PG_DB_PORT='5432'

# Run integration tests
pytest test/store/ -v -m integration
```

**Other Integration Tests:**

- **Qdrant**: Required for Qdrant store tests (optional)
  - Can be run locally or via Docker

Integration tests are marked with `@pytest.mark.integration` and are skipped by default.

## Optional Dependencies

The project has several optional dependency groups:

- `dev`: Development tools (build, twine, ruff, pytest, etc.)
- `testall`: All testing dependencies including PostgreSQL/PGVector and Qdrant
- `pgvector`: PostgreSQL/PGVector support
- `mega`: All optional dependencies (ChromaDB, docx2python, PyPDF2, Streamlit, etc.)

Install with: `pip install ".[testall]"` or `pip install ".[mega]"`

## Additional Resources

- [PyPI Trusted Publishing Guide](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions for Python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)
- [Python Packaging Guide](https://packaging.python.org/en/latest/)
- [OgbujiPT README](README.md) - Main project documentation
- [Demo Examples](demo/) - Example applications and use cases
