from typing import List

import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder

from audioindex.model import Orm

top_k_embedding = 100
top_k_cross_embedding = 10

debug = True


def query(query: str, db_location: str,
          model_name: str = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
          cross_encoder_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
    model = SentenceTransformer(model_name)
    print(f"Query: {query} with model name: {model_name}")

    query_vector = model.encode(query)

    orm = Orm(db_location)
    embeddings = list(orm.get_embeddings(model_name))

    cosine = torch.Tensor([util.cos_sim(query_vector, embedding.embedding) for embedding in embeddings])
    top_results = torch.topk(cosine, k=top_k_embedding)
    original_score = {embeddings[idx].transcribed_paragraph: score_idx for score_idx, idx in enumerate(top_results[1])}

    paragraph_ids = [embeddings[idx].transcribed_paragraph for idx in top_results[1]]

    reevaluate_results(orm, cross_encoder_name, original_score, paragraph_ids, query)


def reevaluate_results(orm: Orm, cross_encoder_name: str, original_scores, paragraph_ids: List[str], query: str):
    paragraphs = [p for p in orm.get_paragraphs(paragraph_ids)]
    cross_encoder = CrossEncoder(cross_encoder_name, max_length=512)
    scores = cross_encoder.predict([[query, paragraph.text] for paragraph in paragraphs], convert_to_tensor=True)
    top_cross_results = torch.topk(scores, k=top_k_cross_embedding)

    transcripts_ids = [paragraphs[idx].transcript_id for idx in top_cross_results[1]]
    transcripts = {t.transcript_id: t for t in orm.get_transcripts(transcripts_ids)}

    for pos, (score, idx) in enumerate(zip(top_cross_results[0], top_cross_results[1])):
        paragraph = paragraphs[idx]
        file_name = transcripts[paragraph.transcript_id].file_name
        pid = paragraph.id
        text = paragraph.text
        if debug:
            print(f"score: {score}, idx: {idx}, original idx: {original_scores[pid]}, pid: {pid}")
        print(f"file {file_name}, start: {paragraph.start} end: {paragraph.end} \n\t'{text}\n\n'")