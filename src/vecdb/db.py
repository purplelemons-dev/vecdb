from hashlib import sha256
from time import time
import os.path
from pickle import dump, load
from typing import Any


class Database:
    def __init__(self, data: dict[str, list[float]] = None):
        self.table = data if data is not None else dict()

    def __repr__(self):
        return f"Database({self.table})"

    @property
    def __dict__(self):
        return self.table

    def to_json(self) -> dict[str, list[float]]:
        # Originally a list comprehension:
        # json_out = {doc: array.tolist() if isinstance(array,ndarray) else array for doc, array in self.table.items()}
        return self.table

    def get(self, key, default=None):
        try:
            return self.table[key]
        except KeyError:
            return default

    def set(self, key, value):
        self.table[key] = value

    def delete(self, key):
        try:
            del self.table[key]
        except KeyError:
            pass

    def save(self, path: str):
        with open(path, "wb") as f:
            dump(self.table, f)


class DatabaseManager:
    def __init__(self, vec_dir: str = "./vecdb"):
        self.databases: dict[str, Database] = dict()
        self.vec_dir = vec_dir.rstrip("/")
        self.__load()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.save()
        for id in self.databases.keys():
            os.rename(f"{self.vec_dir}/{id}.pkl.locked", f"{self.vec_dir}/{id}.pkl")

    def __load(self):
        if not os.path.exists(self.vec_dir):
            os.mkdir(self.vec_dir)

        for filename in os.listdir(self.vec_dir):
            if filename.endswith(".pkl"):
                path = f"{self.vec_dir}/{filename}"

                with open(path, "rb") as f:
                    db_id = filename.split(".")[0]
                    self.databases[db_id] = Database()
                    self.databases[db_id].table = load(f)

                os.rename(path, path + ".locked")

    @property
    def database_names(self):
        return list(self.databases.keys())

    def save(self):
        for id, db in self.databases.items():
            db.save(f"{self.vec_dir}/{id}.pkl.locked")

    def get(self, database_id: str) -> Database:
        try:
            return self.databases[database_id]
        except KeyError as e:
            return None

    def delete(self, database_id: str):
        try:
            del self.databases[database_id]
            os.remove(f"{self.vec_dir}/{database_id}.pkl.locked")
        except KeyError:
            raise ValueError("Database ID does not exist")

    def new(self, database_id: "str|int" = None, data: dict[str, Any] = None):
        "Creates a new database and returns its ID (id can be custom or auto-generated)"

        if database_id is None:
            database_id = sha256(time().hex().encode("utf-8")).hexdigest()[:12]
        if database_id in self.databases:
            raise ValueError("Database ID already exists")

        database_id = str(database_id)

        self.databases[database_id] = Database(data) if data is not None else Database()
        return database_id


if __name__ == "__main__":
    with DatabaseManager("./testdb") as db:
        db.new("test", {"a": [1], "b": 2})
        print(db.get("test"))
        db.get("test").set("c", 3)
        print(db.get("test"))
