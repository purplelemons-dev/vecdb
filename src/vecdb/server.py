"""
Vector database server. Accessible via /api/v1

GET: `/[DB_ID]/vectors` -> `dict[str,Vector]`
    Returns all stored word vectors.

POST: `/create` -> `dict[str,str]`
    Body: `{"id": str, "data": dict[str,Vector]}`
    
    `id` and `data` are optional.
    Creates a new database and returns its ID (id can be custom or auto-generated)
    
POST: `/[DB_ID]/lookup` -> `list[list[dict[str,float]]]`
    Body: `{"topn": int, "words":list[str]}`
    
    Returns a list of topn most similar words for each word in the body and their similarity scores.

POST: `/[DB_ID]/add` -> `dict[str,bool]`
    Body: `list[str]`
    
    Adds the given vectors to the database. Returns a list of booleans indicating whether each vector was already in the database.
"""

from typing_extensions import Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from .embeddings import Embeddings
from .db import DatabaseManager
import json


class Handler(BaseHTTPRequestHandler):
    dbManager: DatabaseManager

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedder: Embeddings

    def parseURL(self):
        url = self.path
        if url.startswith("/api/v1"):
            url = url[7:]
            database_id = url.split("/")[0]
            url = url[len(database_id) + 1 :]
            return database_id, url
        return None, None

    def send_json(self, code: int, obj: dict[Any, Any] | list[Any]):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode("utf-8"))

    @property
    def json(self) -> "dict|list":
        if self.headers["Content-Type"] != "application/json":
            raise ValueError("Expected Content-Type: application/json")
        return json.loads(self.rfile.read(int(self.headers["Content-Length"])))

    def do_GET(self):
        try:
            database_id, api_path = self.parseURL()
            assert (database_id, api_path) != (
                None,
                None,
            ), "Invalid URL. Expected /api/v1/[DB_ID]/[API_CALL]"

            if api_path == "vectors":
                self.send_json(200, self.dbManager.get(database_id).to_json())
            else:
                self.send_response(404, f"Unknown API path: {api_path}")
                self.end_headers()
        except Exception as e:
            print(e)
            self.send_response(500, f"Internal Server Error: {e}")
            self.end_headers()
            return

    def do_POST(self):
        try:
            database_id, api_path = self.parseURL()
            if (database_id, api_path) != (None, None):
                raise ValueError("Invalid URL. Expected /api/v1/[DB_ID]/[API_CALL]")
            assert isinstance(self.json, dict), "Expected JSON body"

            if api_path == "create" and database_id == "":
                database_id = self.json.get("id", None)
                data = self.json.get("data", None)
                database_id = self.dbManager.new(database_id, data)
                self.send_json(200, {"id": database_id})

            elif api_path == "lookup":
                assert "topn" in self.json, "Expected 'topn' field in body"
                assert "words" in self.json, "Expected 'words' field in body"
                topn: int = int(self.json["topn"])
                words: list[str] = self.json["words"]
                embedding = self.embedder(text=words)
                results: dict[str, dict[str,]] = dict()
                for word, vec in embedding.items():
                    temp_closeness = []
                    for db_word, db_vec in self.dbManager.get(
                        database_id
                    ).table.items():
                        temp_closeness.append(
                            {
                                "word": db_word,
                                "closeness": self.embedder.closeness(vec, db_vec),
                            }
                        )
                    temp_closeness.sort(key=lambda x: x["closeness"], reverse=True)
                    results[word] = temp_closeness[:topn]
                self.send_json(200, results)

            elif api_path == "add":
                assert "text" in self.json, "Expected 'text' field in body"
                text = self.json["text"]
                assert isinstance(
                    text, list
                ), "Expected 'text' field to be a list of strings"

                out = dict()
                for doc in text:
                    if self.dbManager.get(database_id).get(doc) is not None:
                        embedding = self.embedder(doc)
                        self.dbManager.get(database_id).set(doc, embedding)
                        out[doc] = True
                    else:
                        out[doc] = False

                self.send_json(200, out)
            else:
                self.send_response(404, f"Unknown API path: {api_path}")
                self.end_headers()

        except Exception as e:
            print(e)
            self.send_response(500, f"Internal Server Error: {e}")
            self.end_headers()
            return


def start_server(
    host: str, port: int, *, dbManager: DatabaseManager, embeddings: Embeddings
):
    server_address = (host, port)
    Handler.dbManager = dbManager
    Handler.embeddings = embeddings
    httpd = HTTPServer(server_address, Handler)
    httpd.serve_forever()
