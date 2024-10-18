To run these tests, first set up a mock Postgres instance with the following commands 
(make sure you don't have anything running on port 0.0.0.0:5432):

```sh
docker pull pgvector/pgvector:pg17
docker run --name mock-postgres -p 5432:5432 \
    -e POSTGRES_USER=mock_user -e POSTGRES_PASSWORD=mock_password -e POSTGRES_DB=mock_db \
    -d pgvector/pgvector:pg17
```

You can also use another PGVector setup, but then you need the following environment variables:

* `PG_HOST`
* `PG_DATABASE`
* `PG_USER`
* `PG_PASSWORD`
* `PG_PORT`

e.g.:

```sh
PG_HOST="localhost"
PG_PORT="5432"
PG_USER="username"
PG_PASSWORD="passwd"
PG_DATABASE="PeeGeeVee"
```

Make sure those are set in the subshell, for example by using an env.sh file and `set -o allexport && source env.sh && set +o allexport`! Further reading: (https://huggingface.co/blog/ucheog/separate-env-setup-from-code)

# Full clean-up/refresh

Sometimes, as dev state evolves, the mock DB container's volume can get messed up. In such cases rebuilding the container won't really address the problem. You need to remove the volume.

```sh
docker volume ls
```

Find the volume name, usually `pgv_db_data` and remove it

```sh
docker volume rm pgv_db_data
```
