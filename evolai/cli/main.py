

import typer
from typing import Optional
from rich.console import Console
from rich import print as rprint

from evolai import __version__
from evolai.cli.commands.miner import miner_app
from evolai.cli.commands.validator import validator_app

console = Console()
err_console = Console(stderr=True, style="bold red")


app = typer.Typer(
    name="evolcli",
    help="EvolAI CLI - Command line interface for EvolAI subnet on Bittensor",
    epilog="Made with ❤️ by EvolAI Team",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def version_callback(value: bool):
    if value:
        console.print(f"EvolCLI version: [bold cyan]{__version__}[/bold cyan]")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit"
    ),
):
    pass


app.add_typer(
    miner_app,
    name="miner",
    help="Miner commands for model registration and validation"
)


app.add_typer(
    validator_app,
    name="validator",
    help="Validator commands for running evaluations and submitting results"
)


@app.command()
def info():
    console.print("\n[bold cyan]EvolAI Subnet Information[/bold cyan]")
    console.print("━" * 60)
    console.print(f"Subnet ID: [bold]47[/bold]")
    console.print(f"Name: [bold]EvolAI[/bold]")
    console.print(f"Focus: [bold]LLM Model Evaluation[/bold]")
    console.print(f"CLI Version: [bold]{__version__}[/bold]")
    console.print("\n[bold]How it works:[/bold]")
    console.print("  • [cyan]Validators[/cyan]: Fetch challenges, evaluate miner models, set on-chain weights")
    console.print("  • [cyan]Miners[/cyan]: Register LLM models trained on the assigned dataset")
    console.print("\n[bold]Competition Tracks:[/bold]")
    console.print("  • [yellow]Transformer Track[/yellow]: Standard transformer models (50% emissions)")
    console.print("  • [yellow]Mamba2 Track[/yellow]: Mamba2/recurrent models (50% emissions)")
    console.print("\n[bold]Commands:[/bold]")
    console.print("  • [green]evolcli miner[/green]     - Register and check model eligibility")
    console.print("  • [green]evolcli validator[/green] - Run evaluation loop and manage validator config")
    console.print("\n[dim]For more information: evolcli --help[/dim]\n")


def cli_main():
    app()


if __name__ == "__main__":
    cli_main()
