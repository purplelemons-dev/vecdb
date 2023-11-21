import vecdb
from .env import Secrets

vecdb.run_database_server(
    vec_dir="./testdb",
    openai_api_key=Secrets.OPENAI_API_KEY,
    openai_org_id=Secrets.OPENAI_ORG_ID,
)
