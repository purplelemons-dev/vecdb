from .db import DatabaseManager
from .server import start_server
from .embeddings import Embeddings


def run_database_server(
    host: str = "127.0.0.1",
    port: int = 6969,
    vec_dir: str = "./vecdb",
    openai_api_key: str = None,
    openai_org_id: str = None,
):
    print(f"Running vecdb server on http://{host}:{port} with {vec_dir=}")
    embedder = Embeddings(openai_api_key, openai_org_id)
    with DatabaseManager(vec_dir) as dbManager:
        start_server(host, port, dbManager=dbManager, embeddings=embedder)


__all__ = ["run_database_server"]
