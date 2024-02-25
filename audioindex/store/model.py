import sqlite3
import struct
import uuid
from dataclasses import dataclass
from typing import Optional, List

import torch


def encode_nparray(arr: torch.Tensor) -> bytes:
    return struct.pack("<" + "f" * len(arr), *arr)


def decode_nparray(blob: bytes) -> torch.Tensor:
    return torch.tensor(list(struct.unpack("<" + "f" * (len(blob) // 4), blob)))


@dataclass
class Transcript:
    transcript_id: str
    status: str
    file_name: str

    def __init__(self, transcript_id: str, status: str, file_name: str):
        self.transcript_id = transcript_id
        self.status = status
        self.file_name = file_name


@dataclass
class TranscribedParagraph:
    id: str
    transcript_id: str
    start: int
    end: int
    text: str
    confidence: Optional[float]
    source: str

    def __init__(self, transcript_id: str, start: int, end: int, text: str, source: str, confidence: Optional[float],
                 pid=None):
        if pid is None:
            pid = uuid.uuid4().hex
        self.id = pid
        self.transcript_id = transcript_id
        self.start = start
        self.end = end
        self.text = text
        self.confidence = confidence
        self.source = source


@dataclass
class Embedding:
    id: uuid.UUID
    transcribed_paragraph: str
    model: str
    embedding: torch.Tensor

    def __init__(self, _id: uuid.UUID, transcribed_paragraph: str, model: str, embedding: torch.Tensor):
        self.id = _id
        self.model = model
        self.embedding = embedding
        self.transcribed_paragraph = transcribed_paragraph


class Orm:
    def __init__(self, db_location: str):
        self.__db_location = db_location
        self.__conn = None

    def __connection(self) -> sqlite3.Connection:
        if self.__conn is None:
            self.__conn = sqlite3.connect(self.__db_location)
            self.__ensure_table_exists(self.__conn, "embeddings", """create table embeddings
            (
                id                    text primary key,
                transcribed_paragraph text not null,
                model                 text not null,
                embedding             blob not null,
                foreign key (transcribed_paragraph) references paragraphs (id)
            )""")
            self.__ensure_table_exists(self.__conn, "paragraphs", """create table paragraphs
            (
                id            text primary key,
                transcript_id text,
                start         int,
                end           int,
                value         text,
                confidence    float,
                source        text,
                foreign key(transcript_id) REFERENCES transcript (id)
            )""")
            self.__ensure_table_exists(self.__conn, "transcript",
                                       "create table transcript(id text primary key, status text, file_name text);")

        return self.__conn

    def __ensure_table_exists(self, con: sqlite3.Connection, table_name: str, create_sql: str):
        cursor = con.cursor()
        cursor.execute(f"select name from sqlite_master where type='table' and name='{table_name}'")
        if cursor.fetchone() is None:
            print(f"Creating table '{table_name}'...")
            cursor.execute(create_sql)
        con.commit()

    def get_transcripts(self, ids: List[str]) -> List[Transcript]:
        cursor = self.__connection().cursor()
        for row in cursor.execute(f"""
                        select t.id, t.status, t.file_name from transcript t
                where t.id in ({','.join(['?'] * len(ids))})
                        """, ids):
            yield Transcript(
                transcript_id=row[0],
                status=row[1],
                file_name=row[2]
            )

    def save_transcripts(self, transcripts: List[Transcript]):
        data = [(transcript.transcript_id, transcript.status, transcript.file_name)
                for transcript in transcripts]
        con = self.__connection()
        cursor = con.cursor()
        cursor.executemany(
            "insert into transcript values (?, ?, ?) on conflict (id) do update set status = excluded.status", data)
        con.commit()

    def get_paragraphs(self, ids: List[str]) -> List[TranscribedParagraph]:
        cursor = self.__connection().cursor()
        for row in cursor.execute(f"""
                select p.id, p.transcript_id, p.start, p.end, p.value, p.confidence, p.source from paragraphs p
        where p.id in ({','.join(['?'] * len(ids))})
                """, ids):
            yield TranscribedParagraph(
                pid=row[0],
                transcript_id=row[1],
                start=row[2],
                end=row[3],
                text=row[4],
                confidence=row[5],
                source=row[6]
            )

    def save_paragraphs(self, paragraphs: List[TranscribedParagraph]):

        data = [(paragraph.id, paragraph.transcript_id, paragraph.start, paragraph.end, paragraph.text,
                 paragraph.confidence, paragraph.source) for
                paragraph in paragraphs]
        con = self.__connection()
        cursor = con.cursor()
        cursor.executemany(
            "insert into paragraphs(id, transcript_id, start, end, value, confidence, source)  values (?, ?, ?, ?, ?, ?, ?)",
            data)
        con.commit()

    def get_pending_paragraphs(self, model: str) -> List[TranscribedParagraph]:
        cursor = self.__connection().cursor()
        for row in cursor.execute("""
        select p.id, p.transcript_id, p.start, p.end, p.value, p.confidence, p.source
from paragraphs p
where p.id not in
      (select distinct(p.id)
       from paragraphs p
                join embeddings e on p.id = e.transcribed_paragraph
       where e.model = ?
          and e.embedding is not null)
        """, (model,)):
            yield TranscribedParagraph(
                pid=row[0],
                transcript_id=row[1],
                start=row[2],
                end=row[3],
                text=row[4],
                confidence=row[5],
                source=row[6]
            )

    def get_embeddings(self, model: str) -> List[Embedding]:
        cursor = self.__connection().cursor()
        for row in cursor.execute("select id, transcribed_paragraph, model, embedding from embeddings where model = ?",
                                  (model,)):
            yield Embedding(
                row[0],
                row[1],
                row[2],
                decode_nparray(row[3])
            )

    def save_embeddings(self, embeddings: List[Embedding]):
        data = [
            (embedding.id.hex, embedding.transcribed_paragraph, embedding.model, encode_nparray(embedding.embedding))
            for
            embedding in embeddings]
        con = self.__connection()
        cursor = con.cursor()
        cursor.executemany(
            "insert into embeddings(id, transcribed_paragraph, model, embedding) values(?, ?, ?, ?)",
            data)
        con.commit()
