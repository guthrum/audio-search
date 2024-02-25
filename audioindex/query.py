from typing import List

import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder

from audioindex.model import generate_embeddings
from audioindex.store.model import Orm
from audioindex.utils import timing

top_k_embedding = 100
top_k_cross_embedding = 10

debug = True


@timing
def query(query: str, db_location: str,
          model_name: str,
          rerank_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
    orm = Orm(db_location)
    embeddings = list(orm.get_embeddings(model_name))
    if not embeddings:
        print(f"No embeddings for model {model_name}, please run embed command first")
        exit(1)

    query_vector = list(generate_embeddings(model_name, [query]))[0]

    top_k_idx = (torch.topk(perform_similarity(embeddings, query_vector), k=top_k_embedding)[1]).cpu().tolist()[0]
    original_score = {embeddings[idx].transcribed_paragraph: score_idx for score_idx, idx in enumerate(top_k_idx)}

    paragraph_ids = [embeddings[idx].transcribed_paragraph for idx in top_k_idx]

    rerank_results_with_cross_encoder(orm, rerank_model, original_score, paragraph_ids, query)


@timing
def perform_similarity(embeddings, query_vector) -> torch.Tensor:
    return util.cos_sim(query_vector, torch.stack([embedding.embedding for embedding in embeddings]))


@timing
def rerank_results_with_cross_encoder(orm: Orm, rerank_model: str, original_scores, paragraph_ids: List[str], query: str):
    cross_encoder = load_cross_encoder(rerank_model)
    paragraphs, top_cross_results, transcripts = reevaluate_impl(cross_encoder, orm, paragraph_ids, query)

    for pos, (score, idx) in enumerate(zip(top_cross_results[0], top_cross_results[1])):
        paragraph = paragraphs[idx]
        file_name = transcripts[paragraph.transcript_id].file_name
        pid = paragraph.id
        text = paragraph.text
        if debug:
            print(f"score: {score}, pos: {pos}, original pos: {original_scores[pid]}, pid: {pid}")
        print(f"file {file_name}, start: {paragraph.start} end: {paragraph.end} \n\t'{text}\n\n'")


@timing
def load_cross_encoder(cross_encoder_name) -> CrossEncoder:
    cross_encoder = CrossEncoder(cross_encoder_name, max_length=512)
    return cross_encoder


@timing
def reevaluate_impl(cross_encoder: CrossEncoder, orm, paragraph_ids, query):
    paragraphs = [p for p in orm.get_paragraphs(paragraph_ids)]
    scores = predict(cross_encoder, paragraphs, query)
    top_cross_results = torch.topk(scores, k=top_k_cross_embedding)
    transcripts_ids = [paragraphs[idx].transcript_id for idx in top_cross_results[1]]
    transcripts = {t.transcript_id: t for t in orm.get_transcripts(transcripts_ids)}
    return paragraphs, top_cross_results, transcripts


@timing
def predict(cross_encoder: CrossEncoder, paragraphs, query):
    scores = cross_encoder.predict([[query, paragraph.text] for paragraph in paragraphs], convert_to_tensor=True)
    return scores
