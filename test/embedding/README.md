To run these tests, first set up a mock Postgres instance with the following commands 
(make sure you don't have anything running on port 0.0.0.0:5432):

```sh
docker pull ankane/pgvector
docker run --name mock-postgres -p 5432:5432 \
    -e POSTGRES_USER=mock_user -e POSTGRES_PASSWORD=mock_password -e POSTGRES_DB=mock_db \
    -d ankane/pgvector
```
