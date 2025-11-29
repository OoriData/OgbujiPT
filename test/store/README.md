Vector Store Tests. By default, tests use **in-memory vector store implementations** (`RAMDataDB`, `RAMMessageDB`), requiring few external dependencies and setup:

```bash
# Run all store tests (fast, no PostgreSQL needed)
pytest test/store/ -v

# Run specific test file
pytest test/store/test_pgvector_data.py -v
```

Tests complete in ~0.5 seconds with zero setup. Perfect for:
- CI/CD pipelines
- Local development without Docker
- Quick iteration during feature development

# In-Memory Vector Stores

These implementations aren't just test doubles - they're **user-facing features** for:
- Rapid prototyping
- Embedded applications
- Demos and tutorials
- Small-scale deployments

See `../../demo/ram-store/README.md` for more info.

# Integration Tests (Optional PostgreSQL)

Integration tests verify behavior against real PostgreSQL with pgvector. These are **skipped by default** but can be run when needed.

### Setup PostgreSQL for Integration Tests

**Option 1: Docker (Recommended)**

```sh
docker pull pgvector/pgvector:pg17
docker run --name mock-postgres -p 5432:5432 \
    -e POSTGRES_USER=mock_user -e POSTGRES_PASSWORD=mock_password -e POSTGRES_DB=mock_db \
    -d pgvector/pgvector:pg17
```

**Option 2: Custom PostgreSQL Setup**

Set these environment variables:

* `PG_DB_HOST` - Database host (default: `localhost`)
* `PG_DB_NAME` - Database name (default: `mock_db`)
* `PG_DB_USER` - Database user (default: `mock_user`)
* `PG_DB_PASSWORD` - Database password (default: `mock_password`)
* `PG_DB_PORT` - Database port (default: `5432`)

Example:

```sh
export PG_DB_HOST='localhost'
export PG_DB_PORT='5432'
export PG_DB_USER='username'
export PG_DB_PASSWORD='passwd'
export PG_DB_NAME='PeeGeeVee'
```

Or use an env.sh file: `set -o allexport && source env.sh && set +o allexport`

Further reading: [Separate env setup from code](https://huggingface.co/blog/ucheog/separate-env-setup-from-code)

### Running Integration Tests

```bash
# Run integration tests with PostgreSQL
pytest test/store/ -v -m integration

# Run all tests (both in-memory and integration)
pytest test/store/ -v -m ""
```

# Full clean-up/refresh

Sometimes, as dev state evolves, the mock DB container's volume can get messed up. In such cases rebuilding the container won't really address the problem. You need to remove the volume.

```sh
docker volume ls
```

Find the volume name, usually `pgv_db_data` and remove it

```sh
docker volume rm pgv_db_data
```
