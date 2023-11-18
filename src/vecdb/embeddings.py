import openai
from .env import Secrets
import numpy as np


class Embeddings:
    "Uses ada v2 to embed the given text"

    def __call__(
        self, text: "str|list[str]", **openai_options
    ) -> "dict[str,np.ndarray]":
        "Uses ada v2 to embed the given text"
        return {
            doc: np.array(embedding.embedding)
            for doc, embedding in zip(
                [text] if isinstance(text, str) else text,
                self.openai_client.embeddings.create(
                    input=text, model="text-embedding-ada-002", **openai_options
                ).data,
            )
        }

    def __init__(self, api_key: str = None, org_id: str = None):
        if api_key is None:
            api_key = Secrets.OPENAI_API_KEY
        if org_id is None:
            org_id = Secrets.OPENAI_ORG_ID

        self.openai_client = openai.OpenAI(api_key=api_key, organization=org_id)

    @staticmethod
    def closeness(
        vec1: "np.ndarray[float]|list[float]", vec2: "np.ndarray[float]|list[float]"
    ) -> float:
        "Returns the closeness score between the two given vectors"

        vec1len, vec2len = len(vec1), len(vec2)
        if vec1len != vec2len:
            raise ValueError(
                f"Cannot compare vectors of different shapes: {vec1len} and {vec2len}"
            )

        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
