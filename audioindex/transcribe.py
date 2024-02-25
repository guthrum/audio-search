import abc
import concurrent.futures
import os
import uuid
from typing import List, Tuple

import assemblyai as aai
from assemblyai import TranscriptStatus

from audioindex.store.model import TranscribedParagraph, Orm, Transcript

API_KEY_ENV_VAR = "ASSEMBLY_AI_API_KEY"


class Transcriber(abc.ABC):
    @abc.abstractmethod
    def transcribe(self, file: str) -> Tuple[Transcript, List[TranscribedParagraph]]:
        pass


class AaiTranscriber(Transcriber):
    def __init__(self):
        if API_KEY_ENV_VAR not in os.environ:
            raise Exception(f"Required environment variable {API_KEY_ENV_VAR} to be set")
        aai.settings.api_key = os.environ[API_KEY_ENV_VAR]
        self.__inner = aai.Transcriber()
        self.__client = aai.Client.get_default()

    def __get_paragraphs(
            self,
            transcript_id: str,
    ) -> aai.types.ParagraphsResponse:
        import httpx
        response = self.__client.http_client.get(
            f"/v2/transcript/{transcript_id}/paragraphs",
        )

        if response.status_code != httpx.codes.ok:
            raise aai.types.TranscriptError(
                f"failed to retrieve paragraphs for transcript {transcript_id}: {response}"
            )

        return aai.types.ParagraphsResponse.parse_obj(response.json())

    def transcribe(self, file: str) -> Tuple[Transcript, List[TranscribedParagraph]]:
        transcript = self.__inner.transcribe(file)
        if transcript.status == TranscriptStatus.completed:
            paragraphs = [
                TranscribedParagraph(transcript.id, paragraph.start, paragraph.end, paragraph.text,
                                     "assemblyai", paragraph.confidence)
                for paragraph in self.__get_paragraphs(transcript_id=transcript.id).paragraphs]
        else:
            paragraphs = []
        return Transcript(transcript_id=transcript.id, status=str(transcript.status), file_name=file), paragraphs


class TestTranscriber(Transcriber):

    def transcribe(self, file: str) -> Tuple[Transcript, List[TranscribedParagraph]]:
        return Transcript(uuid.uuid4().hex, 'completed', file), [
            TranscribedParagraph(file, 0, 100, 'value here', "test", 0.9),
            TranscribedParagraph(file, 101, 200, 'value here 2', "test", 0.3),
        ]


def transcribe(db_location: str, directory: str):
    # transcriber = TestTranscriber()
    transcriber = AaiTranscriber()
    # iterate over directory and pick all files
    file_locations = [directory]
    if os.path.isdir(directory):
        file_locations = [os.path.join(directory, file) for file in os.listdir(directory) if
                          os.path.isfile(os.path.join(directory, file))]
    if not file_locations:
        return

    print(f"uploading files {';'.join(file_locations)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(transcriber.transcribe, file=file): file for file in file_locations}
        # TODO: handle the unfinished ones
        completed, _ = concurrent.futures.wait(futures)
        transcripts: List[Tuple[Transcript, List[TranscribedParagraph]]] = [future.result() for future in completed]

    print("Transcript processing complete, saving to DB")
    orm = Orm(db_location=db_location)
    orm.save_transcripts([transcript[0] for transcript in transcripts])
    orm.save_paragraphs([t for tt in [transcript[1] for transcript in transcripts] for t in tt])
    print("Transcripts saved to DB")
