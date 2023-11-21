import openai
import numpy as np


class Embeddings:
    "Uses ada v2 to embed the given text"

    def __call__(
        self, text: "str|list[str]", **openai_options
    ) -> "dict[str,list[float]]":
        "Uses ada v2 to embed the given text"
        return {
            doc: embedding.embedding
            for doc, embedding in zip(
                [text] if isinstance(text, str) else text,
                self.openai_client.embeddings.create(
                    input=text, model="text-embedding-ada-002", **openai_options
                ).data,
            )
        }

    def __init__(self, api_key: str, org_id: str):
        self.openai_client = openai.OpenAI(api_key=api_key, organization=org_id)

    @staticmethod
    def closeness(vec1: list[float], vec2: list[float]) -> float:
        "Returns the closeness score between the two given vectors"

        vec1len, vec2len = len(vec1), len(vec2)
        if vec1len != vec2len:
            raise ValueError(
                f"Cannot compare vectors of different shapes: {vec1len} and {vec2len}"
            )

        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
