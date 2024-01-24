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
def embed_cmd(db):
    embeddings_audio.index(db)


@click.command('query')
@click.option('--db', required=True, type=click.Path(exists=True, dir_okay=False))
@click.argument("query")
def query_cmd(db, query: str):
    embeddings_query.query(query, db)


cli.add_command(transcribe_cmd)
cli.add_command(embed_cmd)
cli.add_command(query_cmd)

if __name__ == '__main__':
    cli()
