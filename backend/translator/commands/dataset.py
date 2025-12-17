from pathlib import Path

import typer

from translator.preprocess import process_dataset

app = typer.Typer()

VERBOSE: bool = typer.Option(False, "--verbose", help="Show detailed output.")
OUTPUT: Path = typer.Option(
    "data/processed", "-o", "--output", help="Directory to save the processed dataset."
)


@app.command()
def prepare(output: Path = OUTPUT, verbose: bool = VERBOSE) -> None:
    """
    Prepare PJM dataset for training.

    This command processes raw PJM data (images/videos) from data/raw/, extracts hand landmarks,
    and saves the processed dataset to the specified output directory (default: data/processed).

    Example:
        poetry run python -m translator.cli dataset_prepare --output data/processed --verbose
    """
    if verbose:
        typer.echo(f"Starting dataset preparation, saving to {output}...")
    try:
        # Note: Adjust output directory handling if needed, as process_dataset splits into train/val/test
        process_dataset(raw_data_dir="data/raw", output_base_dir="data", num_augmentations=8)
    except Exception as e:
        typer.echo(f"Error during dataset preparation: {str(e)}")
        raise typer.Exit(code=1)
    if verbose:
        typer.echo("Dataset preparation completed.")
