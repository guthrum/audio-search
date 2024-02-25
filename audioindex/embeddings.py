import itertools
import uuid
from typing import Tuple, Iterator

from audioindex.model import generate_embeddings
from audioindex.store.model import Orm, Embedding


def batched(iterable, n):
    # since we are using python 3.11
    # https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, n)):
        yield batch


def index(db_location: str, model_name: str):
    orm = Orm(db_location)
    pending_embeddings = [(paragraph.text, paragraph.id) for paragraph in orm.get_pending_paragraphs(model_name)]
    if pending_embeddings:
        for to_save in batched(run_model(model_name, pending_embeddings), 96):
            print(f"Saving {len(to_save)} embeddings to DB")
            orm.save_embeddings(to_save)
    print(f"All paragraphs embedded for model {model_name}")


def run_model(model_name: str, pending_embeddings: list[Tuple[str, str]]) -> Iterator[Embedding]:
    embeddings = generate_embeddings(model_name, [e[0] for e in pending_embeddings])
    for (transcript, embedding) in zip([e[1] for e in pending_embeddings], embeddings):
        yield Embedding(uuid.uuid4(), transcript, model_name, embedding)
