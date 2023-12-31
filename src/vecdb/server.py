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

from typing import Any
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
            sets = url[8:].split("/")
            database_id = sets[0]
            api_path = None
            if len(sets) > 1:
                api_path = "/".join(sets[1:])
            return database_id, api_path
        return None, None

    def send_text(self, code: int, text: str, content_type: str = "text/plain"):
        try:
            text_data = text.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "keep-alive")
            self.send_header("Content-Length", len(text_data))
            self.end_headers()
            self.wfile.write(text_data)
        except Exception:
            raise
        return

    def send_json(self, code: int, obj: Any):
        self.send_text(code, json.dumps(obj,default=lambda o: o.__dict__), content_type="application/json")

    def do_GET(self):
        if self.path == "/":
            self.send_json(200, {"message": "Hello World!"})
            return
        if not self.path.startswith("/api/v1"):
            self.send_response(404, "Invalid URL. Expected /api/v1/[DB_ID]/[API_CALL]")
            self.end_headers()
            return
        try:
            database_id, api_path = self.parseURL()

            db = self.dbManager.get(database_id)
            if api_path == "vectors" and db is not None:
                db_json = db.to_json()
                for k, v in db_json.items():
                    print(f"{k=}, {type(k)=}")
                    print(f"{type(v)=}")
                    print(f"{isinstance(v, list)=}\n")
                #self.send_json(200, db)
                self.send_text(200, json.dumps(db,default=lambda o: o.__dict__), content_type="application/json")
                return
            elif database_id == "databases":
                self.send_json(200, self.dbManager.database_names)
                return
            else:
                self.send_response(404, f"Unknown API path: {api_path}")
                self.end_headers()
                return
        except Exception as e:
            print(e)
            self.send_response(500, f"Internal Server Error: {e}")
            self.end_headers()
            raise

    def do_POST(self):
        if not self.path.startswith("/api/v1"):
            self.send_response(404, "Invalid URL. Expected /api/v1/[DB_ID]/[API_CALL]")
            self.end_headers()
            return
        json_data = self.rfile.read(int(self.headers["Content-Length"]))
        json_data = json.loads(json_data)
        try:
            database_id, api_path = self.parseURL()
            if database_id is None:
                raise ValueError(
                    f"Invalid URL {database_id=}, {api_path=}. Expected /api/v1/[DB_ID]/[API_CALL]"
                )
            assert isinstance(json_data, dict), "Expected JSON body"

            if database_id == "create" and api_path is None:
                database_id: "str|None" = json_data.get("id", None)
                data: "dict|None" = json_data.get("data", None)
                database_id = self.dbManager.new(database_id, data)
                self.send_json(200, {"id": database_id})
                return

            elif api_path == "lookup":
                topn: int = int(json_data.get("topn",1))
                words: list[str] = json_data.get("words")
                if words is None:
                    raise ValueError("Expected 'words' field in body")
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
                assert "text" in json_data, "Expected 'text' field in body"
                text = json_data["text"]
                assert isinstance(
                    text, list
                ), "Expected 'text' field to be a list of strings"

                out = dict()
                for doc in text:
                    print(f"doing {doc}")
                    db = self.dbManager.get(database_id)
                    if db is not None:
                        vec = db.get(doc)
                        if vec is not None:
                            out[doc] = True
                            continue
                        else:
                            embedding = self.embedder(doc)[doc]
                            self.dbManager.get(database_id).set(doc, embedding)
                            out[doc] = False

                self.send_json(200, out)
            else:
                self.send_response(404, f"Unknown API path: {api_path}")
                self.end_headers()

        except Exception as e:
            print(e)
            self.send_response(500, f"Internal Server Error: {e}")
            self.end_headers()
            raise


def start_server(
    host: str, port: int, *, dbManager: DatabaseManager, embeddings: Embeddings
):
    server_address = (host, port)
    Handler.dbManager = dbManager
    Handler.embedder = embeddings
    httpd = HTTPServer(server_address, Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()
        print("Server Stopped")
