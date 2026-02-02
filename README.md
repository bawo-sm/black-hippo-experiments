# Black Hippo product classification
- python app coded in the FastAPI framework
- contains several endpoints responisble for product classification, color recognition and HS codes classification


# Structure
- `notebooks` - working sketches and presentations
- `resources` - working local data
- `src` - source code of the python app
    - `ai/` - module with classes responsible for ai actions
    - `services/` - module with classes responsible for single services, e.g. sql db connection
    - `endpoints/` - moduel with classes responsible for endpoints logic
    - `common/` - module with scripts used in different places, e.g. data schemas
    - `main.py` - the main FastApi app script
    - `settings.py` - settings and parameters for the app algorithms


# Environmental variables
```txt
# azure
STORAGE_ACCOUNT_KEY="..."
ACCOUNT_NAME="..."

# azure sql
SQL_ENDPOINT="..."
SQL_DATABASE="..."
SQL_USERNAME="..."
SQL_PASSWORD="..."
SQL_PORT="..."

# azure openai
OPENAI_ENDPOINT="..."
OPENAI_KEY="..."

# vecto db qdrant
QDRANT_URL="..."
QDRANT_API_KEY="..."
```


# Local lanuch
- create and activate python virtual env, e.g.
```
python3 -m venv venv
source venv/bin/activate
```
- install requirements:
```
pip install uv
uv pip instal -r requriements.txt
```
- the command `uvicorn src.main:app` runs the FastAPI app
- see endpoints swagger on [/docs](http://localhost:8000/docs)
- see endpoints structure on [/openapi.json](http://localhost:8000/openapi.json)