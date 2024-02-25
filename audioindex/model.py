import itertools
from typing import Iterator


def __batched(iterable, n):
    # since we are using python 3.11
    # https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, n)):
        yield batch


def generate_embeddings(model_name: str, content: list[str]) -> Iterator[str]:
    if model_name.startswith("cohere"):
        return cohere_embeddings(model_name.removeprefix("cohere/"), content)
    else:
        return sentence_transformer_embedding(model_name, content)


def sentence_transformer_embedding(model_name: str, content: list[str]) -> Iterator[str]:
    from sentence_transformers import SentenceTransformer
    print(f"Loading model {model_name}")
    model = SentenceTransformer(model_name)
    print("Starting encoding....")
    return model.encode(content, convert_to_tensor=True).tolist()


def cohere_embeddings(model_name: str, content: list[str]) -> Iterator[str]:
    import cohere
    import os

    cohere_api_key = os.environ['COHERE_API_KEY']
    if not cohere_api_key:
        print("require cohere api key to be set via environment variable COHERE_API_KEY")
        exit(1)
    print(f"Running embeddings with cohere model {model_name}")
    co = cohere.Client(cohere_api_key)

    # batch pending embeddings into sublists of 96 for the API
    for (idx, batch) in enumerate(__batched(content, 96)):
        print(f"Making request {idx} size {len(batch)}")
        co_response = co.embed(
            texts=batch,
            model=model_name,
            input_type="search_document"
        )
        for embedding in co_response.embeddings:
            yield embedding
