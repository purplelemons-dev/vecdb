import argparse
from . import run_database_server

parser = argparse.ArgumentParser(description="VecDB")

parser.add_argument(
    "--host", type=str, default="127.0.0.1", help="Host IP address to bind to"
)
parser.add_argument("--port", type=int, default=6969, help="Host port to bind to")
parser.add_argument(
    "--dir", type=str, default="./vecdb", help="Path to database directory"
)
parser.add_argument(
    "--keyfile",
    type=str,
    help="Path to key file with openai api key and org id. key=value format",
)

args = parser.parse_args()

try:
    with open(args.keyfile) as f:
        key, org = map(lambda x: x.split("=")[1].strip(), f.read().split("\n")[:2])
except FileNotFoundError:
    raise ValueError(f"Could not find keyfile at {args.keyfile}")


run_database_server(
    host=args.host,
    port=args.port,
    vec_dir=args.dir,
    openai_api_key=key,
    openai_org_id=org,
)
