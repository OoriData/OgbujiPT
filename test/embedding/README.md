To run these tests, first set up a mock Postgres instance with the following commands 
(make sure you don't have anything running on port 0.0.0.0:5432):

```sh
docker pull ankane/pgvector
docker run --name mock-postgres -p 5432:5432 \
    -e POSTGRES_USER=mock_user -e POSTGRES_PASSWORD=mock_password -e POSTGRES_DB=mock_db \
    -d ankane/pgvector
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
