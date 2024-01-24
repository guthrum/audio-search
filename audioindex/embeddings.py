import uuid

import torch
from sentence_transformers import SentenceTransformer

from audioindex.model import Orm, Embedding


def info():
    print("cuda available: " + torch.cuda.is_available())
    print("cuda device count: " + torch.cuda.device_count())
    print("cuda device name: " + torch.cuda.get_device_name())


def index(db_location: str, model_name: str = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'):
    orm = Orm(db_location)
    pending_embeddings = [(paragraph.text, paragraph.id) for paragraph in orm.get_pending_paragraphs(model_name)]

    print(f"Loading model {model_name}")
    model = SentenceTransformer(model_name)

    print("Starting encoding....")
    embeddings = model.encode([s[0] for s in pending_embeddings], convert_to_tensor=True)
    to_save = [Embedding(uuid.uuid4(), transcript, model_name, embedding) for (transcript, embedding) in
               zip([e[1] for e in pending_embeddings], embeddings)]
    orm.save_embeddings(to_save)
