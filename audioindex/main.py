import click
from audioindex import transcribe as transcribe_audio
from audioindex import embeddings as embeddings_audio
from audioindex import query as embeddings_query


@click.group()
def cli():
    pass


@click.command('transcribe')
@click.option('--file', '-f', required=True, type=click.Path(exists=True))
@click.option('--db', required=True, type=click.Path(exists=False, dir_okay=False))
def transcribe_cmd(file, db):
    transcribe_audio.transcribe(db, file)


@click.command('embed')
@click.option('--db', required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('--model', default="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
def embed_cmd(db, model: str):
    embeddings_audio.index(db, model_name=model)


@click.command('query')
@click.option('--db', required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('--model', default="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
@click.argument("query")
def query_cmd(db, query: str, model: str):
    embeddings_query.query(query, db, model_name=model)


cli.add_command(transcribe_cmd)
cli.add_command(embed_cmd)
cli.add_command(query_cmd)

if __name__ == '__main__':
    cli()
