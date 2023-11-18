from .db import DatabaseManager
from .server import start_server
from .embeddings import Embeddings


def run_database_server(
    host: str,
    port: int,
    vec_dir: str = "./vecdb",
    openai_api_key: str = None,
    openai_org_id: str = None,
):
    embedder = Embeddings(openai_api_key, openai_org_id)
    with DatabaseManager(vec_dir) as dbManager:
        start_server(host, port, dbManager=dbManager, embeddings=embedder)


__all__ = ["run_database_server"]
