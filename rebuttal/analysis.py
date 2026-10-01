import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    from rdn.experiment import Result
    from rdn import viz

    return (Path,)


@app.cell
def _(Path):
    PATH_TO_PAR_FOLDER = Path(__file__).parent / "parameters"
    PATH_TO_SAVE_FOLDER = Path(__file__).parent / "output"
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
