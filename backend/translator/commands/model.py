import typer

from translator.tf import PJMClassifier

app = typer.Typer()

VERBOSE: bool = typer.Option(False, "--verbose", help="Show detailed output.")
EPOCHS: int = typer.Option(50, "--epochs", help="Number of training epochs.")
BATCH_SIZE: int = typer.Option(32, "--batch-size", help="Batch size for training.")


@app.command()
def train(
    epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, verbose: bool = VERBOSE
) -> None:
    """
    Train TensorFlow model for PJM recognition.

    This command trains a TensorFlow model using the processed dataset from data/train/.
    The trained model will be saved to models/.
    """
    if verbose:
        logging.debug("Verbose mode enabled.")
        typer.echo("Starting model training with TensorFlow...")
    try:
        classifier = PJMClassifier()
        classifier.train(epochs=epochs, batch_size=batch_size)
    except Exception as e:
        typer.echo(f"Error during model training: {str(e)}")
        raise typer.Exit(code=1)
    if verbose:
        typer.echo("Model training completed.")


@app.command()
def verify(verbose: bool = VERBOSE) -> None:
    """
    Verify trained model performance.

    This command verifies the trained TensorFlow model's accuracy using the test dataset
    from data/test/.

    Example:
        poetry run python -m translator.cli model_verify --verbose
    """
    if verbose:
        typer.echo("Starting model verification...")
    try:
        classifier = PJMClassifier()
        loss, accuracy = classifier.evaluate()
        typer.echo(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    except Exception as e:
        typer.echo(f"Error during model verification: {str(e)}")
        raise typer.Exit(code=1)
    if verbose:
        typer.echo("Model verification completed.")
