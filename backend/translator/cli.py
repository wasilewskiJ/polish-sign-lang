# backend/translator/cli.py
import importlib.metadata
import os
import typer

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_CPP_MAX_VLOG_LEVEL", "0")
os.environ.setdefault("GLOG_minloglevel", "3")
from translator.commands import dataset_app, model_app, webcam_app

app = typer.Typer(
    name="pjm2jp-translator",
    help="CLI tool for managing PJM Translator tasks, including dataset preparation, model training, and verification.",
    add_completion=False,
)
app.add_typer(
    model_app, name="model", help="Commands for training and verifying the model."
)
app.add_typer(dataset_app, name="dataset", help="Commands for preparing the dataset.")
app.add_typer(
    webcam_app, name="webcam", help="Commands for testing mediapipe with webcam."
)


@app.command()
def version() -> None:
    typer.echo(f"PJM2JP CLI Version: {importlib.metadata.version('pjm2jp')}")


if __name__ == "__main__":
    app()
