import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def doctor() -> None:
    """Print detected runtime info and exit."""
    raise NotImplementedError("biota doctor lands in M1")
