
"""
Validator CLI commands for EvolAI subnet

Commands:
- setup: Verify validator environment
- get-miners: List registered miners from the Bittensor chain
- config: Manage validator configuration
- list-local: View local evaluation results
- run: Continuous loss-based evaluation loop with auto weight setting
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
console = Console()
err_console = Console(stderr=True, style="bold red")


validator_app = typer.Typer(
    name="validator",
    help="Validator commands for running evaluations and submitting results",
    no_args_is_help=True
)

_NUM_QUESTIONS = 1


@validator_app.command("setup")
def setup_check():
    """
    Verify that the validator environment is correctly configured.

    Checks:
    - Python version
    - Core packages (bittensor, torch, transformers)
    - Optional vLLM binary reachable (not required for current loss-based evaluation)
    - GPU availability & VRAM
    - .env file present
    """
    from dotenv import load_dotenv
    load_dotenv()
    import shutil
    import subprocess as _sp

    console.print("\n[bold cyan]EvolAI Validator — Environment Check[/bold cyan]\n")

    ok_count = 0
    warn_count = 0
    fail_count = 0

    def ok(msg):
        nonlocal ok_count; ok_count += 1
        console.print(f"  [green]✓[/green] {msg}")

    def warn(msg):
        nonlocal warn_count; warn_count += 1
        console.print(f"  [yellow]⚠[/yellow] {msg}")

    def fail(msg):
        nonlocal fail_count; fail_count += 1
        console.print(f"  [red]✗[/red] {msg}")


    import sys
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 9):
        ok(f"Python {py_ver}")
    else:
        fail(f"Python {py_ver} — need >= 3.9")


    for pkg in ["bittensor", "torch", "transformers", "openai", "httpx", "rich"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            ok(f"{pkg} {ver}")
        except ImportError:
            fail(f"{pkg} not installed")


    vllm_bin = os.environ.get("VLLM_EXECUTABLE", "vllm")
    vllm_path = shutil.which(vllm_bin)
    if vllm_path:
        try:
            result = _sp.run([vllm_path, "--version"], capture_output=True, text=True, timeout=10)
            ver = result.stdout.strip() or result.stderr.strip() or "unknown"
            ok(f"Optional vLLM binary: {vllm_path} ({ver})")
        except Exception:
            ok(f"Optional vLLM binary: {vllm_path}")
    elif vllm_bin != "vllm":
        warn(f"Optional vLLM binary not found at: {vllm_bin}")
    else:
        warn(
            "vLLM binary not on PATH. This is OK for the current HuggingFace "
            "loss evaluation flow."
        )


    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            gpus = []
            for i in range(n):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                gpus.append(f"GPU {i}: {name} ({mem:.0f} GB)")
            ok(f"{n} GPU(s) detected")
            for g in gpus:
                console.print(f"      {g}")

            if n >= 4:
                console.print(
                    "    [dim]Recommended: --parallel-miners 3 "
                    "(judge GPU 0, miners GPU 1-3)[/dim]"
                )
            elif n >= 2:
                console.print(
                    "    [dim]Recommended: default layout "
                    "(judge GPU 0, miner GPU 1)[/dim]"
                )
        else:
            warn("No CUDA GPUs detected — torch.cuda.is_available() = False")
    except ImportError:
        fail("torch not installed — cannot check GPUs")


    env_file = Path(".env")
    if env_file.exists():
        ok(f".env file present ({env_file.resolve()})")
    else:
        warn(".env file not found — run [dim]bash scripts/setup-validator.sh[/dim] to create one")


    console.print()
    parts = [f"[green]{ok_count} passed[/green]"]
    if warn_count:
        parts.append(f"[yellow]{warn_count} warnings[/yellow]")
    if fail_count:
        parts.append(f"[red]{fail_count} failed[/red]")
    console.print(f"  {', '.join(parts)}\n")

    if fail_count:
        raise typer.Exit(1)


@validator_app.command("get-miners")
def get_miners(
    track: Optional[str] = typer.Option(
        None,
        "--track",
        help="Filter by track: transformer, mamba2, or all"
    ),
    netuid: int = typer.Option(
        47,
        "--netuid",
        help="Subnet netuid (default: 47)"
    ),
    network: str = typer.Option(
        "finney",
        "--network",
        help="Bittensor network: finney, test, or local"
    )
):
    """
    Fetch registered miners directly from Bittensor chain commitment metadata

    Reads on-chain commitment data for every UID in the subnet metagraph,
    decodes the compressed model metadata, and displays a summary table.

    Shows:
    - Miner UID and hotkey
    - Registered models for each track (transformer / mamba2)
    - Model revisions
    """
    from dotenv import load_dotenv
    load_dotenv()
    from evolai.utils.metadata import decompress_metadata


    if track is None:
        track = Prompt.ask(
            "Select track",
            choices=["transformer", "mamba2", "all"],
            default="all"
        )


    track = track.lower()
    if track in ["both", "all"]:
        track = None


    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        progress.add_task(f"Connecting to Bittensor ({network})...", total=None)
        try:
            import bittensor as bt
            subtensor = bt.Subtensor(network=network)
            metagraph = subtensor.metagraph(netuid)
        except Exception as e:
            err_console.print(f"\n❌ Failed to connect to Bittensor: {e}")
            raise typer.Exit(code=1)

    console.print(f"[green]✓[/green] Connected — subnet [bold]{netuid}[/bold], [bold]{len(metagraph.hotkeys)}[/bold] UIDs")

    miners = []
    uids_no_meta = []


    try:
        from huggingface_hub import HfApi as _HfApi
        _hf_api = _HfApi()
    except Exception:
        _hf_api = None

    def _get_hf_upload_time(model_name, revision):
        if not _hf_api or not model_name or model_name == '-':
            return None
        try:
            repo_info = _hf_api.repo_info(model_name, revision=revision, repo_type="model")
            commits = list(_hf_api.list_repo_commits(model_name, repo_type="model"))
            revision_to_check = revision or repo_info.sha
            for commit in commits:
                if commit.commit_id == revision_to_check or commit.commit_id.startswith(revision_to_check or ""):
                    return commit.created_at
            return repo_info.last_modified
        except Exception:
            return None


    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Scanning chain commitments...", total=len(metagraph.hotkeys))

        for uid in range(len(metagraph.hotkeys)):
            hotkey = metagraph.hotkeys[uid]
            progress.update(task, advance=1, description=f"Scanning UID {uid}/{len(metagraph.hotkeys) - 1}…")

            try:
                commit_data = subtensor.get_commitment_metadata(netuid, hotkey)
                if not commit_data:
                    uids_no_meta.append(uid)
                    continue


                if not (isinstance(commit_data, dict) and 'info' in commit_data):
                    uids_no_meta.append(uid)
                    continue

                fields = commit_data['info']['fields']
                if not (fields and len(fields) > 0 and fields[0] and len(fields[0]) > 0):
                    uids_no_meta.append(uid)
                    continue

                raw_data = fields[0][0]

                raw_key = next((k for k in raw_data if k.startswith('Raw') and k[3:].isdigit()), None)
                if raw_key is None:
                    uids_no_meta.append(uid)
                    continue

                compressed_bytes = bytes(raw_data[raw_key][0])
                metadata = decompress_metadata(compressed_bytes)
                if not metadata:
                    uids_no_meta.append(uid)
                    continue

                transformer_info = metadata.get('transformer', {})
                mamba2_info = metadata.get('mamba2', {})
                has_transformer = bool(transformer_info.get('model_name'))
                has_mamba2 = bool(mamba2_info.get('model_name'))


                if track == 'transformer' and not has_transformer:
                    continue
                if track == 'mamba2' and not has_mamba2:
                    continue
                if not has_transformer and not has_mamba2:
                    uids_no_meta.append(uid)
                    continue

                miners.append({
                    'uid': uid,
                    'hotkey': hotkey,
                    'metadata': metadata,
                    't_upload': _get_hf_upload_time(transformer_info.get('model_name'), transformer_info.get('revision')) if has_transformer else None,
                    'm_upload': _get_hf_upload_time(mamba2_info.get('model_name'), mamba2_info.get('revision')) if has_mamba2 else None,
                })

            except Exception:
                uids_no_meta.append(uid)


    if not miners:
        console.print(f"\n[yellow]No miners registered{' for ' + track + ' track' if track else ''}[/yellow]\n")
        return

    table = Table(
        title=f"Registered Miners — Subnet {netuid}{' (' + track.upper() + ' track)' if track else ''}",
        show_header=True,
        header_style="bold cyan"
    )
    table.add_column("UID", style="dim", justify="right")
    table.add_column("Hotkey", style="bold")
    table.add_column("Transformer Model")
    table.add_column("T-Rev", style="dim")
    table.add_column("T-Upload", style="dim")
    table.add_column("Mamba2 Model")
    table.add_column("M-Rev", style="dim")
    table.add_column("M-Upload", style="dim")

    for miner in miners:
        uid = miner['uid']
        hotkey = miner['hotkey']
        meta = miner['metadata']

        transformer_info = meta.get('transformer', {})
        mamba2_info = meta.get('mamba2', {})

        t_model = transformer_info.get('model_name', '-')
        t_rev = str(transformer_info.get('revision', '-') or 'main')
        t_up = miner.get('t_upload')
        t_up_str = t_up.strftime('%Y-%m-%d') if t_up else '-'

        m_model = mamba2_info.get('model_name', '-')
        m_rev = str(mamba2_info.get('revision', '-') or 'main')
        m_up = miner.get('m_upload')
        m_up_str = m_up.strftime('%Y-%m-%d') if m_up else '-'

        table.add_row(
            str(uid),
            hotkey[:16] + "…" if len(hotkey) > 16 else hotkey,
            t_model,
            t_rev[:10] + "…" if len(t_rev) > 10 else t_rev,
            t_up_str,
            m_model,
            m_rev[:10] + "…" if len(m_rev) > 10 else m_rev,
            m_up_str,
        )

    console.print("\n")
    console.print(table)
    console.print(f"\n[dim]Total: {len(miners)} miners | UIDs without metadata: {len(uids_no_meta)}[/dim]\n")


@validator_app.command("config")
def validator_config(
    show: Optional[bool] = typer.Option(
        False,
        "--show",
        help="Show current configuration"
    ),
    set_key: Optional[str] = typer.Option(
        None,
        "--set",
        help="Set configuration key (format: key=value)"
    )
):
    """
    Show or update validator configuration.

    Available keys:
    - use_wandb     true/false — enable Weights & Biases run logging
    - wandb_project string    — W&B project name (default: evol-validator)

    Examples:
        evolcli validator config --show
        evolcli validator config --set use_wandb=true
        evolcli validator config --set wandb_project=my-project
    """
    from dotenv import load_dotenv
    load_dotenv()
    config_dir = Path.home() / ".evolai" / "validator"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.json"


    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'use_wandb': False,
            'wandb_project': 'evol-validator'
        }


    if set_key:
        if '=' not in set_key:
            err_console.print("\n❌ Invalid format. Use: --set key=value")
            raise typer.Exit(code=1)

        key, value = set_key.split('=', 1)
        key = key.strip()
        value = value.strip()


        if key == 'use_wandb':
            config[key] = value.lower() in ['true', '1', 'yes']
        else:
            config[key] = value


        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        console.print(f"\n[green]✓ Configuration updated:[/green] {key} = {config[key]}\n")


    if show or set_key:
        console.print(Panel.fit(
            f"[bold]Validator Configuration[/bold]\n\n"
            f"W&B Logging: [cyan]{config.get('use_wandb', False)}[/cyan]\n"
            f"W&B Project: [cyan]{config.get('wandb_project', 'N/A')}[/cyan]\n\n"
            f"[dim]Config file: {config_file}[/dim]",
            title="Configuration",
            border_style="cyan"
        ))
        console.print()


@validator_app.command("list-local")
def list_local_results(
    track: Optional[str] = typer.Option(
        None,
        "--track",
        help="Filter by track"
    ),
    limit: Optional[int] = typer.Option(
        10,
        "--limit",
        help="Number of results to show"
    )
):
    """
    List locally saved evaluation results.

    Reads JSON files written to ~/.evolai/validator/results/ and
    prints a summary table (timestamp, track, miner count, top scores).

    Examples:
        evolcli validator list-local
        evolcli validator list-local --track transformer --limit 5
    """
    results_dir = Path.home() / ".evolai" / "validator" / "results"
    
    if not results_dir.exists():
        console.print("\n[yellow]No local results found[/yellow]\n")
        return
    

    result_files = sorted(results_dir.glob("evaluation_*.json"), reverse=True)
    
    if not result_files:
        console.print("\n[yellow]No evaluation results found[/yellow]\n")
        return
    

    if track:
        result_files = [f for f in result_files if track in f.name]
    

    result_files = result_files[:limit]
    
    console.print(f"\n[bold]Local Evaluation Results[/bold] ([dim]showing {len(result_files)}[/dim])\n")
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            console.print(f"[cyan]━[/cyan] {result_file.name}")
            console.print(f"   Track: {data.get('track', 'N/A')} | "
                        f"Miners: {len(data.get('results', []))} | "
                        f"Questions: {data.get('num_questions', 'N/A')} | "
                        f"Timestamp: {data.get('timestamp', 'N/A')[:19]}")
            

            results = data.get('results', [])
            if results:
                sorted_results = sorted(results, key=lambda x: x.get('raw_score', 0), reverse=True)
                console.print("   [dim]Top scores:[/dim]", end=" ")
                for i, r in enumerate(sorted_results[:3], 1):
                    console.print(f"UID {r.get('miner_uid')}: {r.get('raw_score', 0):.2f}", end="  ")
                console.print()
            console.print()
            
        except Exception as e:
            console.print(f"[red]Error reading {result_file.name}: {e}[/red]\n")
    
    console.print(f"[dim]Results directory: {results_dir}[/dim]\n")


def _scan_miners_from_chain(
    subtensor,
    netuid: int,
    eval_track: str,
    console,
    verbose: bool = False,
    subtensor_lock=None,
) -> tuple:
    """
    Scan all UIDs in the subnet metagraph for on-chain commitment metadata.

    Returns:
        (miners, uids_without_metadata)
        miners: list of dicts with uid, hotkey, coldkey, model_name, revision, metadata
    """
    from evolai.utils.metadata import decompress_metadata

    import contextlib as _ctx
    _lock_ctx = subtensor_lock if subtensor_lock is not None else _ctx.nullcontext()

    with _lock_ctx:
        metagraph = subtensor.metagraph(netuid)
    miners: list = []
    uids_without_metadata: list = []

    for uid in range(len(metagraph.hotkeys)):
        hotkey = metagraph.hotkeys[uid]
        coldkey = metagraph.coldkeys[uid] if hasattr(metagraph, 'coldkeys') else ""
        try:
            with _lock_ctx:
                commit_data = subtensor.get_commitment_metadata(netuid, hotkey)
            if not commit_data:
                uids_without_metadata.append(uid)
                continue

            if not (isinstance(commit_data, dict) and 'info' in commit_data):
                uids_without_metadata.append(uid)
                continue

            fields = commit_data['info']['fields']
            if not (fields and len(fields) > 0 and fields[0] and len(fields[0]) > 0):
                uids_without_metadata.append(uid)
                continue

            raw_data = fields[0][0]

            raw_key = next((k for k in raw_data if k.startswith('Raw') and k[3:].isdigit()), None)
            if raw_key is None:
                if verbose:
                    console.print(f"  [yellow]UID {uid}: No RawN field in commitment data[/yellow]")
                uids_without_metadata.append(uid)
                continue

            compressed_bytes = bytes(raw_data[raw_key][0])
            metadata = decompress_metadata(compressed_bytes)
            if not metadata:
                if verbose:
                    console.print(f"  [yellow]UID {uid}: metadata is None after decompression[/yellow]")
                uids_without_metadata.append(uid)
                continue

            if verbose:
                console.print(f"  [dim]UID {uid}: metadata={metadata}[/dim]")

            track_info = metadata.get(eval_track, {})
            if track_info and track_info.get('model_name'):
                miners.append({
                    'uid': uid,
                    'hotkey': hotkey,
                    'coldkey': coldkey,
                    'model_name': track_info['model_name'],
                    'revision': track_info.get('revision', 'main'),
                    'metadata': metadata,
                })
                if verbose:
                    console.print(f"  [green]UID {uid}: Found {eval_track} miner: {track_info['model_name']}[/green]")
            else:
                if verbose:
                    console.print(f"  [yellow]UID {uid}: No {eval_track} track in metadata[/yellow]")
                uids_without_metadata.append(uid)

        except Exception as e:
            if verbose:
                console.print(f"  [red]UID {uid}: Error reading metadata: {e}[/red]")
            uids_without_metadata.append(uid)

    return miners, uids_without_metadata


@validator_app.command("run")
def run_validator(
    wallet_name: Optional[str] = typer.Option(
        None,
        "--wallet",
        help="Validator wallet name"
    ),
    hotkey_name: Optional[str] = typer.Option(
        None,
        "--hotkey",
        help="Validator hotkey name"
    ),
    netuid: Optional[int] = typer.Option(
        47,
        "--netuid",
        help="Subnet UID (default: 47)"
    ),
    num_questions: Optional[int] = typer.Option(
        _NUM_QUESTIONS,
        "--questions",
        help="Number of questions per evaluation"
    ),
    eval_interval: Optional[int] = typer.Option(
        300,
        "--eval-interval",
        help="Seconds between evaluations (default: 300 = 5 min)"
    ),
    weight_interval: Optional[int] = typer.Option(
        1800,
        "--weight-interval",
        help="Seconds between weight updates (default: 1800 = 30 min)"
    ),
    use_wandb: Optional[bool] = typer.Option(
        True,
        "--wandb/--no-wandb",
        help="Enable Weights & Biases logging"
    ),
    wandb_project: Optional[str] = typer.Option(
        "evol-validator",
        "--wandb-project",
        help="W&B project name"
    ),
    debug: Optional[bool] = typer.Option(
        False,
        "--debug",
        help="Debug mode",
        hidden=True,
    ),
    fake_wallet: Optional[bool] = typer.Option(
        False,
        "--fake-wallet",
        help="Testing mode",
        hidden=True,
    ),
    vllm_bin: Optional[str] = typer.Option(
        None,
        "--vllm-bin",
        help="Internal option",
        hidden=True,
    ),
    track: Optional[str] = typer.Option(
        None,
        "--track",
        help="Evaluate only this track: transformer or mamba2 (default: both)",
    ),
):
    """
    Run the validator loop.
    """
    from dotenv import load_dotenv
    load_dotenv()
    import time
    import logging
    import threading

    if wallet_name is None:
        wallet_name = Prompt.ask("Validator wallet name", default="default")

    if hotkey_name is None:
        hotkey_name = Prompt.ask("Validator hotkey name", default="default")


    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    for _noisy in (
        "httpx", "httpcore", "openai._base_client",
        "urllib3", "asyncio", "bittensor", "websockets",
    ):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    if vllm_bin:
        os.environ["VLLM_EXECUTABLE"] = vllm_bin

    console.print("\n[bold cyan]Initializing Validator...[/bold cyan]\n")
    
    if fake_wallet:

        import hashlib
        import secrets
        

        fake_seed = secrets.token_hex(16)
        validator_coldkey = "5" + hashlib.sha256(f"cold_{fake_seed}".encode()).hexdigest()[:47]
        validator_hotkey = "5" + hashlib.sha256(f"hot_{fake_seed}".encode()).hexdigest()[:47]
        
        console.print(f"[yellow]⚠ Using FAKE wallet for testing[/yellow]")
        console.print(f"  Fake Coldkey: {validator_coldkey}")
        console.print(f"  Fake Hotkey: {validator_hotkey}\n")
    else:
        with console.status("[cyan]Loading wallet..."):
            try:
                from bittensor_wallet import Wallet
                wallet = Wallet(name=wallet_name, hotkey=hotkey_name)
                
                if not wallet.hotkey_file.exists_on_device():
                    err_console.print(f"\n❌ Hotkey '{hotkey_name}' not found")
                    raise typer.Exit(code=1)
                
                validator_coldkey = wallet.coldkeypub.ss58_address
                validator_hotkey = wallet.hotkeypub.ss58_address
                
            except Exception as e:
                err_console.print(f"\n❌ Failed to load wallet: {e}")
                raise typer.Exit(code=1)
        
        console.print(f"[green]✓ Wallet loaded[/green]")
        console.print(f"  Coldkey: {validator_coldkey}")
        console.print(f"  Hotkey: {validator_hotkey}\n")


    if fake_wallet:
        validator_auth = None
    else:
        from evolai.validator.challenge_client import ValidatorAuth as _ValidatorAuth
        _wallet_ref = wallet
        validator_auth = _ValidatorAuth(
            hotkey=validator_hotkey,
            sign_fn=lambda msg: _wallet_ref.hotkey.sign(msg.encode()).hex(),
        )
    

    with console.status("[cyan]Connecting to Bittensor..."):
        try:
            import bittensor as bt
            subtensor = bt.Subtensor()
            metagraph = subtensor.metagraph(netuid=netuid)
            console.print(f"[green]✓ Connected to Bittensor[/green] (netuid={netuid}, neurons={len(metagraph.uids)})\n")
        except Exception as e:
            err_console.print(f"\n❌ Failed to connect to Bittensor: {e}")
            raise typer.Exit(code=1)
    

    console.print()
    

    wandb_run = None
    if use_wandb:
        try:
            import wandb

            wandb_entity = os.getenv("WANDB_ENTITY", "open-evolai")
            wandb_project = (wandb_project or "evol-validator").strip()
            wandb_api_key = os.getenv("WANDB_API_KEY", "").strip()

            if wandb_api_key:
                wandb.login(key=wandb_api_key, relogin=True)
            else:


                if not wandb.login(anonymous="never"):
                    console.print(
                        "[yellow]⚠ W&B login not found, logging disabled[/yellow]"
                    )
                    console.print(
                        f"[dim]Set WANDB_API_KEY or run `wandb login` to log runs to {wandb_entity}/{wandb_project}[/dim]\n"
                    )
                    use_wandb = False

            if use_wandb:
                run_name = f"validator_{validator_coldkey}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

                wandb_run = wandb.init(
                    entity=wandb_entity,
                    project=wandb_project,
                    name=run_name,
                    config={
                        'validator_coldkey': validator_coldkey,
                        'validator_hotkey': validator_hotkey,
                        'netuid': netuid,
                        'tracks': ['transformer', 'mamba2'],
                        'eval_interval': eval_interval,
                        'weight_interval': weight_interval
                    },
                    tags=["validator", f"netuid_{netuid}", "transformer", "mamba2", "loss-eval"]
                )
                console.print(f"[green]✓ W&B initialized[/green] | Run: {run_name}\n")
        except ImportError:
            console.print("[yellow]⚠ wandb not installed, logging disabled[/yellow]")
            console.print("[dim]Install with: pip install wandb[/dim]\n")
            use_wandb = False
        except Exception as e:
            console.print(
                f"[yellow]⚠ W&B initialization failed for {wandb_entity}/{wandb_project}: {e}[/yellow]\n"
            )
            use_wandb = False
    

    from evolai.validator.evaluator import ModelValidator
    from evolai.validator.epoch_manager import (
        generate_seed,
        commit_epoch_seed,
        epoch_eval_order,
        derive_indices,
        current_epoch as _current_epoch,
        EVAL_SALT,
    )
    from evolai.validator.progress_tracker import ProgressTracker
    from evolai.validator.challenge_client import (
        fetch_challenge_texts,
        submit_evaluations,
        submit_weights,
    )
    from evolai.validator.config import (
        EPOCH_BLOCKS,
        N_EVAL,
        W_ABS,
        W_FLOW,
        W_QUALITY,
        PROGRESS_GAMMA,
        PROGRESS_EMA_ALPHA,
        HISTORY_EPOCHS,
        PROGRESS_MIN_EVALUATIONS,
        DAILY_ALPHA_EMISSION,
        DAILY_TAO_EMISSION,
        STAGNATION_BURN_UID,
        OWNER_API_URL,
        get_eval_config_for_model_size,
        ACTIVE_DATASETS,
        DATASET_SIZES,
        EVAL_REFERENCE_TOKENIZER,
        EVAL_THINK_MAX_NEW_TOKENS,
        EVAL_THINK_MAX_NEW_TOKENS_MAMBA2,
        MIN_FLOW_EPOCHS,
        FLOW_EPS,
        EMISSION_LAMBDA,
        EMA_SHORT_ROUNDS,
        EMA_LONG_ROUNDS,
        W_IMPROVEMENT,
        W_PROXIMITY,
        EMISSION_FLOOR,
        EMISSION_PROXIMITY_THRESHOLD,
        EVAL_PENALTY_LOSS,
        COLDKEY_ARCHIVE_TTL_DAYS,
        WEIGHT_EXPONENT,
        SIDE_QUEST_MAX_CTX,
        SIDE_QUEST_MAX_NEW_TOKENS,
    )
    from evolai.validator.loss_evaluator import (
        compute_cross_entropy_loss,
        evaluate_with_side_quests,
        ChatSample as _ChatSample,
    )
    import random as _random
    import time as _time
    import torch as _torch
    from rich.progress import (
        Progress, BarColumn, TaskProgressColumn,
        TimeElapsedColumn, TextColumn, MofNCompleteColumn,
    )


    progress_tracker = ProgressTracker(
        w_abs=W_ABS,
        w_flow=W_FLOW,
        w_quality=W_QUALITY,
        gamma=PROGRESS_GAMMA,
        ema_alpha=PROGRESS_EMA_ALPHA,
        history_epochs=HISTORY_EPOCHS,
        min_evaluations=PROGRESS_MIN_EVALUATIONS,
        min_flow_epochs=MIN_FLOW_EPOCHS,
        flow_eps=FLOW_EPS,
        emission_lambda=EMISSION_LAMBDA,
        ema_short_rounds=EMA_SHORT_ROUNDS,
        ema_long_rounds=EMA_LONG_ROUNDS,
        w_improvement=W_IMPROVEMENT,
        w_proximity=W_PROXIMITY,
        emission_floor=EMISSION_FLOOR,
        emission_proximity_threshold=EMISSION_PROXIMITY_THRESHOLD,
        archive_ttl_days=COLDKEY_ARCHIVE_TTL_DAYS,
    )
    owner_api = OWNER_API_URL


    console.print(
        f"[cyan]Loading reference tokenizer: {EVAL_REFERENCE_TOKENIZER}…[/cyan]"
    )
    from transformers import AutoTokenizer as _AutoTokenizer
    _ref_tokenizer = _AutoTokenizer.from_pretrained(
        EVAL_REFERENCE_TOKENIZER, trust_remote_code=False,
    )
    console.print(
        f"[green]✓ Reference tokenizer loaded[/green] "
        f"(vocab_size={_ref_tokenizer.vocab_size})\n"
    )

    active_tracks = ("transformer", "mamba2")
    if track:
        _t = track.lower()
        if _t not in active_tracks:
            err_console.print(f"\n❌ Unknown track '{track}'. Choose: transformer, mamba2")
            raise typer.Exit(code=1)
        active_tracks = (_t,)

    def _get_track_score_maps(
        min_evaluations: int,
        subtensor_lock=None,
    ) -> Dict[str, Dict[int, float]]:
        track_scores: Dict[str, Dict[int, float]] = {}
        for track_name in active_tracks:
            try:
                track_miners, _ = _scan_miners_from_chain(
                    subtensor, netuid, track_name, console, verbose=False,
                    subtensor_lock=subtensor_lock,
                )
                track_uids = [miner['uid'] for miner in track_miners]
                # Compute per-track global best long EMA fresh each round.
                global_best = progress_tracker.compute_global_best_long_ema(
                    uids=track_uids
                )
                all_scores = progress_tracker.get_all_scores(
                    global_best_long_ema=global_best
                )
                track_scores[track_name] = {
                    uid: score for uid, score in all_scores.items()
                    if uid in set(track_uids)
                }
            except Exception as exc:
                logging.warning(
                    f"[weights] Failed to load {track_name} miners: {exc}"
                )
                track_scores[track_name] = {}
        return track_scores
    
    console.print(Panel.fit(
        f"[bold]Validator Configuration[/bold]\n\n"
        f"Netuid: [cyan]{netuid}[/cyan]\n"
        f"Tracks: [cyan]transformer, mamba2[/cyan]\n"
        f"Eval Mode: [cyan]Epoch-based decentralised[/cyan]\n"
        f"Epoch Blocks: [cyan]{EPOCH_BLOCKS}[/cyan] (~{EPOCH_BLOCKS * 12 // 60} min/epoch)\n"
        f"Eval rows per dataset: [cyan]{N_EVAL}[/cyan]\n"
        f"Active datasets: [cyan]{', '.join(ACTIVE_DATASETS)}[/cyan]\n"
        f"Score weights: [cyan]abs={W_ABS} flow={W_FLOW} quality={W_QUALITY}[/cyan]\n"
        f"EMA alpha: [cyan]{PROGRESS_EMA_ALPHA}[/cyan]  Min flow epochs: [cyan]{MIN_FLOW_EPOCHS}[/cyan]  Emission λ: [cyan]{EMISSION_LAMBDA}[/cyan]\n"
        f"History epochs: [cyan]{HISTORY_EPOCHS}[/cyan]\n"
        f"W&B Logging: [cyan]{use_wandb}[/cyan]\n"
        f"W&B Project: [cyan]{wandb_project if use_wandb else 'N/A'}[/cyan]\n"
        f"Owner API (telemetry only): [cyan]{owner_api}[/cyan]",
        title="Starting Validator Loop",
        border_style="green"
    ))
    
    if not Confirm.ask("\n[yellow]Start continuous evaluation loop?[/yellow]"):
        console.print("[dim]Cancelled[/dim]")
        raise typer.Exit(code=0)

    console.print("\n[bold green]Validator loop started[/bold green] (Press Ctrl+C to stop)\n")
    console.print("━" * 80 + "\n")


    _ds_size_cache: Dict[str, int] = dict(DATASET_SIZES)

    def _get_dataset_size(dataset_name: str) -> Optional[int]:
        """Return dataset row count from cache, then try HuggingFace."""
        if dataset_name in _ds_size_cache:
            return _ds_size_cache[dataset_name]
        try:
            from datasets import load_dataset_builder as _load_builder
            builder = _load_builder(dataset_name)
            splits = builder.info.splits or {}
            size = None
            for split_name in ("train", "validation", "test"):
                if split_name in splits:
                    size = splits[split_name].num_examples
                    break
            if size:
                _ds_size_cache[dataset_name] = size
                logging.info(f"Dataset {dataset_name!r}: {size} rows (fetched from HF)")
                return size
        except Exception as exc:
            logging.warning(f"Could not fetch size for {dataset_name!r}: {exc}")
        return None


    def _find_my_uid() -> int:
        for _uid_i in range(len(metagraph.hotkeys)):
            if metagraph.hotkeys[_uid_i] == validator_hotkey:
                return _uid_i
        return -1


    _model_cache: dict = {"key": None, "model": None, "tokenizer": None, "cleanup_fn": None}

    def _evict_model_cache() -> None:
        """Release the cached model from GPU memory and purge its disk cache."""
        import gc
        import torch
        from evolai.validator.evaluator import purge_hf_model_cache as _purge
        cached_key = _model_cache["key"]
        if cached_key is None:
            return
        if _model_cache["cleanup_fn"] is not None:
            try:
                _model_cache["cleanup_fn"]()
            except Exception:
                pass
        _model_cache.update({"key": None, "model": None, "tokenizer": None, "cleanup_fn": None})
        torch.cuda.empty_cache()
        gc.collect()

        cached_model_name = cached_key.split("@")[0] if cached_key else None
        if cached_model_name:
            try:
                _purge(cached_model_name)
            except Exception:
                pass
        if cached_key:
            console.print(f"  [dim]Evicted cached model: {cached_key}[/dim]")


    import signal as _signal
    _stop_requested = threading.Event()
    # Substrate websocket is single-threaded; serialize all subtensor calls.
    _subtensor_lock = threading.Lock()

    def _request_stop(signum, frame):
        console.print(
            f"\n[yellow]Received signal {signum} — stopping after current round[/yellow]"
        )
        _stop_requested.set()

    try:
        _signal.signal(_signal.SIGTERM, _request_stop)
    except (OSError, AttributeError):
        pass

    _WEIGHT_INTERVAL_S = 600

    def _weight_update_worker() -> None:
        import time as _wt_time
        _wt_time.sleep(60)
        while not _stop_requested.is_set():
            _t0 = _wt_time.time()
            try:
                if fake_wallet:
                    console.print("[dim][weights] disabled (fake wallet)[/dim]")
                else:
                    _all_uids = list(range(256))
                    _wts = [0.0] * 256
                    _burn_uid = (
                        STAGNATION_BURN_UID
                        if STAGNATION_BURN_UID is not None
                        else 0
                    )
                    _alpha_price = 0.0

                    # Collect all evaluated UIDs across all tracks for the
                    # global emission scale computation.
                    _all_eval_uids = list(progress_tracker.get_all_scores().keys())
                    _emission_scale = progress_tracker.compute_emission_scale(
                        uids=_all_eval_uids
                    )
                    _stale_days = progress_tracker.get_staleness_days()
                    console.print(
                        f"  [cyan][weights] emission scale={_emission_scale:.3f}"
                        + (f" (no improving miners, staleness={_stale_days:.1f}d)"
                           if _emission_scale < 1.0 else " (active miners)") + "[/cyan]"
                    )
                    if use_wandb and wandb_run:
                        wandb_run.log({
                            "emission_scale": _emission_scale,
                            "staleness_days": _stale_days,
                        })

                    try:
                        with _subtensor_lock:
                            if hasattr(subtensor, "get_subnet_price"):
                                _ap = subtensor.get_subnet_price(netuid)
                                _alpha_price = (
                                    float(_ap.tao)
                                    if hasattr(_ap, "tao")
                                    else float(_ap)
                                )
                            else:
                                _ap_rao = subtensor.substrate.runtime_call(
                                    api="SwapRuntimeApi",
                                    method="current_alpha_price",
                                    params=[netuid],
                                ).value
                                _alpha_price = float(_ap_rao) / 1e9
                    except Exception as _ap_err:
                        logging.warning(
                            f"[weights] alpha price fetch failed: {_ap_err} — "
                            f"defaulting budget to 1.0"
                        )
                        _alpha_price = -1.0  # sentinel: skip budget cap

                    if _alpha_price > 0:
                        _miner_budget = min(
                            1.0,
                            DAILY_TAO_EMISSION / (_alpha_price * DAILY_ALPHA_EMISSION),
                        )
                    else:
                        # price unavailable or zero — don't penalise miners
                        _miner_budget = 1.0
                    console.print(
                        f"  [weights] alpha={max(0.0, _alpha_price):.6f} TAO/α  "
                        f"budget={_miner_budget:.6f}"
                    )

                    _track_scores_w = _get_track_score_maps(
                        min_evaluations=PROGRESS_MIN_EVALUATIONS,
                        subtensor_lock=_subtensor_lock,
                    )
                    _active_tracks_w = [
                        (n, s) for n, s in _track_scores_w.items() if s
                    ]
                    _share = (
                        1.0 / len(_active_tracks_w)
                        if _active_tracks_w
                        else 0.0
                    )

                    for _tn, _ts in _active_tracks_w:
                        _exp = {
                            u: sc ** WEIGHT_EXPONENT
                            for u, sc in _ts.items()
                        }
                        _tot = sum(_exp.values())
                        if _tot <= 0:
                            continue
                        for _uw, _ew in _exp.items():
                            if 0 <= _uw < 256:
                                _wts[_uw] += _share * (_ew / _tot)

                    _effective_budget = _miner_budget * _emission_scale
                    for _i in range(256):
                        if _i != _burn_uid:
                            _wts[_i] *= _effective_budget
                    _wts[_burn_uid] = max(0.0, 1.0 - _effective_budget)

                    with _subtensor_lock:
                        _sw_result = subtensor.set_weights(
                            netuid=netuid,
                            wallet=wallet,
                            uids=_all_uids,
                            weights=_wts,
                            wait_for_inclusion=True,
                            wait_for_finalization=False,
                        )
                    if isinstance(_sw_result, tuple) and len(_sw_result) == 2:
                        _sw_ok, _sw_err = _sw_result
                    elif isinstance(_sw_result, bool):
                        _sw_ok, _sw_err = _sw_result, None
                    else:
                        _sw_ok = bool(_sw_result) if _sw_result is not None else False
                        _sw_err = None

                    if _sw_ok:
                        console.print(
                            f"  [green]✓ [weights] set at "
                            f"{datetime.utcnow().strftime('%H:%M:%S UTC')}[/green]"
                        )
                        if use_wandb and wandb_run:
                            wandb_run.log({"weight_update_success": 1})
                        # Send the exact weight vector to the owner proxy,
                        # normalized to sum to 1.0 — matching Bittensor's
                        # on-chain normalization.
                        try:
                            with _subtensor_lock:
                                _cur_block = subtensor.get_current_block()
                        except Exception:
                            _cur_block = None
                        _wts_sum = sum(_wts)
                        _wts_normalized = (
                            [w / _wts_sum for w in _wts]
                            if _wts_sum > 0.0
                            else _wts
                        )
                        submit_weights(
                            validator_hotkey=validator_hotkey,
                            netuid=netuid,
                            weights={i: w for i, w in enumerate(_wts_normalized)},
                            block=_cur_block,
                            owner_api_url=owner_api,
                            auth=validator_auth,
                        )
                    else:
                        console.print(
                            f"  [red]✗ [weights] set_weights failed: {_sw_err}[/red]"
                        )
                        if use_wandb and wandb_run:
                            wandb_run.log({"weight_update_failure": 1})

            except Exception as _wt_err:
                console.print(
                    f"  [red]✗ [weights] error: {_wt_err}[/red]"
                )
                if use_wandb and wandb_run:
                    wandb_run.log({"weight_update_error": 1})

            _elapsed = _wt_time.time() - _t0
            _stop_requested.wait(timeout=max(1.0, _WEIGHT_INTERVAL_S - _elapsed))

    _weight_thread = threading.Thread(
        target=_weight_update_worker, daemon=True, name="weight-updater"
    )
    _weight_thread.start()

    # ------------------------------------------------------------------ #
    # Model prefetch pipeline                                             #
    # A background thread downloads upcoming miner models to disk while  #
    # the current one is being evaluated.  At most _MAX_PREFETCH_SLOTS   #
    # directories are kept on disk at once (1 active + 2 ahead).         #
    # ------------------------------------------------------------------ #
    import queue as _pf_q

    _PREFETCH_BASE = Path.home() / ".evolai" / "prefetch_cache"
    _PREFETCH_BASE.mkdir(parents=True, exist_ok=True)
    # Wipe any stale dirs left from a previous crashed run.
    import shutil as _shutil_pf
    for _stale in list(_PREFETCH_BASE.iterdir()):
        try:
            _shutil_pf.rmtree(_stale, ignore_errors=True)
        except Exception:
            pass

    _MAX_PREFETCH_SLOTS = 3   # 1 being evaluated + 2 lookahead
    _N_PREFETCH_AHEAD   = 2

    _prefetch_lock   = threading.Lock()
    _prefetch_ready: dict = {}   # cache_key → Path (fully downloaded)
    _prefetch_active: set = set()  # currently downloading
    _prefetch_errors: set = set()  # failed this epoch – skip retry
    _prefetch_q: _pf_q.Queue = _pf_q.Queue()

    def _prefetch_worker() -> None:
        import shutil as _sh
        from evolai.validator.evaluator import prefetch_model_to_disk as _pf
        while not _stop_requested.is_set():
            try:
                _ck, _mn, _rv, _dd = _prefetch_q.get(timeout=2)
            except _pf_q.Empty:
                continue
            try:
                console.print(f"  [dim][prefetch] Downloading {_ck}…[/dim]")
                _pf(_mn, _rv, str(_dd))
                with _prefetch_lock:
                    _prefetch_active.discard(_ck)
                    _prefetch_ready[_ck] = _dd
                console.print(f"  [dim][prefetch] Ready: {_ck}[/dim]")
            except Exception as _pf_err:
                console.print(
                    f"  [dim][prefetch] Failed {_ck}: {_pf_err}[/dim]"
                )
                with _prefetch_lock:
                    _prefetch_active.discard(_ck)
                    _prefetch_errors.add(_ck)
                if _dd.exists():
                    _sh.rmtree(_dd, ignore_errors=True)

    def _enqueue_prefetch(cache_key: str, model_name: str, revision: str) -> None:
        with _prefetch_lock:
            if (
                cache_key in _prefetch_ready
                or cache_key in _prefetch_active
                or cache_key in _prefetch_errors
            ):
                return
            # Evict oldest ready slot if we'd exceed the disk budget.
            while len(_prefetch_ready) + len(_prefetch_active) >= _MAX_PREFETCH_SLOTS:
                if not _prefetch_ready:
                    break  # all slots are active downloads; just let it queue
                import shutil as _sh_ev
                _old_ck, _old_dd = next(iter(_prefetch_ready.items()))
                del _prefetch_ready[_old_ck]
                if _old_dd.exists():
                    _sh_ev.rmtree(_old_dd, ignore_errors=True)
            _dest = _PREFETCH_BASE / cache_key.replace("/", "_").replace("@", "__at__")
            _prefetch_active.add(cache_key)
        _prefetch_q.put((cache_key, model_name, revision, _dest))

    def _pop_prefetch_dir(cache_key: str):
        """Return and remove the prefetch Path for cache_key if ready, else None."""
        with _prefetch_lock:
            return _prefetch_ready.pop(cache_key, None)

    def _discard_prefetch_dir(dest_dir) -> None:
        if dest_dir is not None:
            import shutil as _sh_cl
            try:
                if dest_dir.exists():
                    _sh_cl.rmtree(dest_dir, ignore_errors=True)
            except Exception:
                pass

    _prefetch_thread = threading.Thread(
        target=_prefetch_worker, daemon=True, name="model-prefetch"
    )
    _prefetch_thread.start()

    MAX_CRASH_RESTARTS = 10
    CRASH_BACKOFF_BASE_S = 30
    _run_start = datetime.utcnow()
    crash_count = 0
    epoch_count = 0
    last_committed_epoch = -1
    current_epoch_seed: Optional[str] = None

    while not _stop_requested.is_set():
        try:
            while not _stop_requested.is_set():
                epoch_count += 1
                epoch_start = datetime.utcnow()


                try:
                    with _subtensor_lock:
                        current_block = subtensor.get_current_block()
                except Exception as _blk_err:
                    logging.warning(f"[run] get_current_block failed: {_blk_err}")
                    _stop_requested.wait(timeout=60)
                    continue

                epoch_num = _current_epoch(current_block, EPOCH_BLOCKS)
                blocks_in_epoch = current_block % EPOCH_BLOCKS
                blocks_remaining = EPOCH_BLOCKS - blocks_in_epoch

                console.print(
                    f"[bold cyan]━━━ Epoch #{epoch_num} (Loop #{epoch_count}) ━━━[/bold cyan]"
                    f" block={current_block},"
                    f" ~{blocks_remaining * 12 // 60}m remaining\n"
                )


                if epoch_num != last_committed_epoch:
                    current_epoch_seed = generate_seed()
                    if not fake_wallet:
                        with _subtensor_lock:
                            _seed_ok, _seed_err = commit_epoch_seed(
                                wallet, subtensor, netuid, epoch_num, current_epoch_seed
                            )
                        if _seed_ok:
                            console.print(
                                f"  [green]✓ Committed epoch seed[/green] "
                                f"epoch={epoch_num} seed={current_epoch_seed[:8]}…"
                            )
                        else:
                            console.print(
                                f"  [yellow]⚠ Seed commit failed — "
                                f"using local seed[/yellow]"
                            )
                            console.print(
                                f"    [dim]Reason: {_seed_err}[/dim]"
                            )
                    else:
                        console.print(
                            f"  [dim]Fake wallet: epoch seed not committed "
                            f"epoch={epoch_num} seed={current_epoch_seed[:8]}…[/dim]"
                        )
                    last_committed_epoch = epoch_num


                try:
                    with _subtensor_lock:
                        metagraph = subtensor.metagraph(netuid=netuid)
                    for _uid_i in range(len(metagraph.hotkeys)):
                        _coldkey_i = (
                            metagraph.coldkeys[_uid_i]
                            if hasattr(metagraph, "coldkeys")
                            and _uid_i < len(metagraph.coldkeys)
                            else ""
                        )
                        replaced = progress_tracker.sync_uid(
                            _uid_i,
                            metagraph.hotkeys[_uid_i],
                            coldkey=_coldkey_i,
                        )
                        if replaced:
                            console.print(
                                f"  [yellow]⚠ UID {_uid_i}: new miner — "
                                f"history wiped[/yellow]"
                            )
                except Exception as _mg_err:
                    logging.warning(f"[run] Metagraph refresh failed: {_mg_err}")
                    console.print(
                        "  [yellow]⚠ Metagraph refresh failed — "
                        "using cached state[/yellow]"
                    )


                my_uid = _find_my_uid()

                for eval_track in active_tracks:
                    console.print(
                        f"[bold]Evaluating {eval_track.upper()} track…[/bold]\n"
                    )

                    try:
                        miners, _ = _scan_miners_from_chain(
                            subtensor, netuid, eval_track, console, verbose=debug,
                            subtensor_lock=_subtensor_lock,
                        )
                        console.print(
                            f"  Found [bold]{len(miners)}[/bold] {eval_track} miners"
                        )

                        if not miners:
                            console.print(
                                f"  [yellow]⚠ No miners for {eval_track} — skipping[/yellow]\n"
                            )
                            continue

                        miner_uids = [m['uid'] for m in miners]
                        uid_to_miner = {m['uid']: m for m in miners}
                        ordered_uids = epoch_eval_order(
                            validator_hotkey, epoch_num, miner_uids
                        )

                        # Build a flat plan used for look-ahead prefetch.
                        _eval_plan = [
                            (
                                _u,
                                uid_to_miner[_u]['model_name'],
                                uid_to_miner[_u].get('revision') or 'main',
                                "{mn}@{rv}".format(
                                    mn=uid_to_miner[_u]['model_name'],
                                    rv=uid_to_miner[_u].get('revision') or 'main',
                                ),
                            )
                            for _u in ordered_uids
                        ]
                        # Reset per-epoch error set and seed the first N slots.
                        _prefetch_errors.clear()
                        for _pj in range(min(_N_PREFETCH_AHEAD, len(_eval_plan))):
                            _, _pf_mn, _pf_rv, _pf_ck = _eval_plan[_pj]
                            _enqueue_prefetch(_pf_ck, _pf_mn, _pf_rv)

                        round_results = []
                        skipped = []

                        for i, uid in enumerate(ordered_uids, 1):
                            if _stop_requested.is_set():
                                break

                            miner = uid_to_miner[uid]
                            hotkey: str = miner['hotkey']
                            model_name: str = miner['model_name']
                            revision: str = miner.get('revision') or 'main'

                            console.print(
                                f"  [{i}/{len(ordered_uids)}] UID {uid} | "
                                f"{model_name} @ {revision} | hotkey {hotkey[:12]}…"
                            )


                            try:
                                chain_hotkey = metagraph.hotkeys[uid]
                                if chain_hotkey != hotkey:
                                    console.print(
                                        f"    [yellow]⚠ hotkey mismatch — skipping[/yellow]"
                                    )
                                    skipped.append({'uid': uid, 'reason': 'hotkey mismatch'})
                                    continue
                            except IndexError:
                                pass


                            _model_obj = _tok = None
                            cleanup_fn = None
                            model_validator = None
                            _using_cache = False
                            _store_in_cache = False
                            cache_key = f"{model_name}@{revision}"
                            datasets_for_eval: Dict[str, list] = {}

                            # Slide the prefetch window: queue the model that
                            # is _N_PREFETCH_AHEAD steps after this one.
                            _next_pj = (i - 1) + _N_PREFETCH_AHEAD
                            if _next_pj < len(_eval_plan):
                                _, _pf_mn, _pf_rv, _pf_ck = _eval_plan[_next_pj]
                                _enqueue_prefetch(_pf_ck, _pf_mn, _pf_rv)
                            _prefetch_dir = _pop_prefetch_dir(cache_key)

                            try:

                                for ds_name in ACTIVE_DATASETS:
                                    ds_size = _get_dataset_size(ds_name)
                                    if ds_size is None:
                                        logging.warning(
                                            f"Unknown size for {ds_name!r} — skipping"
                                        )
                                        continue
                                    indices = derive_indices(
                                        seed=current_epoch_seed,
                                        uid=uid,
                                        dataset_name=ds_name,
                                        dataset_size=ds_size,
                                        n=N_EVAL,
                                        salt=EVAL_SALT,
                                    )
                                    datasets_for_eval[ds_name] = indices

                                if not datasets_for_eval:
                                    console.print(
                                        f"    [yellow]⚠ No datasets available — skipping[/yellow]"
                                    )
                                    skipped.append({'uid': uid, 'reason': 'no datasets'})
                                    continue

                                _dataset_summary = ", ".join(
                                    f"{n}({len(idx)} idx)"
                                    for n, idx in datasets_for_eval.items()
                                )
                                console.print(f"    Challenge: {_dataset_summary}")


                                _texts = fetch_challenge_texts(datasets_for_eval)
                                if not _texts:
                                    console.print(
                                        f"    [yellow]⚠ Could not fetch texts — skipping[/yellow]"
                                    )
                                    skipped.append({'uid': uid, 'reason': 'texts unavailable'})
                                    continue

                                _total_indices = sum(
                                    len(v) for v in datasets_for_eval.values()
                                )
                                console.print(
                                    f"    Fetched {len(_texts)} texts "
                                    f"({_total_indices} indices)"
                                )


                                if (
                                    _model_cache["key"] == cache_key
                                    and _model_cache["model"] is not None
                                ):
                                    _model_obj = _model_cache["model"]
                                    _tok = _model_cache["tokenizer"]
                                    _using_cache = True
                                    _store_in_cache = True
                                    console.print(f"    [dim]Reusing loaded model[/dim]")
                                    _discard_prefetch_dir(_prefetch_dir)
                                    _prefetch_dir = None
                                else:
                                    if _model_cache["key"] is not None:
                                        _evict_model_cache()
                                    model_validator = ModelValidator()
                                    _load_label = (
                                        "local prefetch"
                                        if _prefetch_dir is not None
                                        else "HF download"
                                    )
                                    with console.status(
                                        f"    [cyan]Loading model ({_load_label})…[/cyan]"
                                    ):
                                        # On non-bf16 GPUs (e.g. V100), Mamba2 SSM
                                        # models trained in bfloat16 overflow when
                                        # cast to float16 (exp/softplus → inf → NaN).
                                        # Force float32 for the mamba2 track on such
                                        # hardware — sub-4B models fit easily.
                                        _force_fp32 = (
                                            eval_track == "mamba2"
                                            and not _torch.cuda.is_bf16_supported()
                                        )
                                        _model_obj, _tok, is_vllm, cleanup_fn = (
                                            model_validator.load_model(
                                                model_name, revision, use_vllm=False,
                                                prefetch_dir=(
                                                    str(_prefetch_dir)
                                                    if _prefetch_dir is not None
                                                    else None
                                                ),
                                                force_float32=_force_fp32,
                                            )
                                        )
                                    console.print(f"    Loaded ({_load_label})")
                                    _discard_prefetch_dir(_prefetch_dir)
                                    _prefetch_dir = None
                                    _store_in_cache = True


                                _num_params_b = (
                                    sum(p.numel() for p in _model_obj.parameters()) / 1e9
                                )
                                if _num_params_b >= 4.0:
                                    console.print(
                                        f"    [yellow]⚠ Model too large "
                                        f"({_num_params_b:.2f}B ≥ 4B) — skipping[/yellow]"
                                    )
                                    skipped.append({
                                        'uid': uid,
                                        'reason': f'model too large ({_num_params_b:.2f}B)',
                                    })
                                    _store_in_cache = False
                                    continue


                                if hasattr(_model_obj, "config") and hasattr(
                                    _model_obj.config, "vocab_size"
                                ):
                                    _model_vocab = _model_obj.config.vocab_size
                                    _ref_vocab = len(_ref_tokenizer)
                                    if _model_vocab < _ref_vocab:
                                        console.print(
                                            f"    [yellow]⚠ Vocab incompatible "
                                            f"(model={_model_vocab} < "
                                            f"ref={_ref_vocab}) — skipping[/yellow]"
                                        )
                                        skipped.append({
                                            'uid': uid,
                                            'reason': (
                                                f'vocab incompatible '
                                                f'({_model_vocab} < {_ref_vocab})'
                                            ),
                                        })
                                        _store_in_cache = False
                                        continue


                                _eval_batch, _eval_max_seq = get_eval_config_for_model_size(
                                    _num_params_b
                                )
                                console.print(
                                    f"    Model {_num_params_b:.2f}B → "
                                    f"batch={_eval_batch}, seq={_eval_max_seq}"
                                )


                                _device = "cuda" if _torch.cuda.is_available() else "cpu"
                                _eval_start = _time.monotonic()

                                _chat_samples = [
                                    t for t in _texts if isinstance(t, _ChatSample)
                                ]
                                _plain_texts = [t for t in _texts if isinstance(t, str)]


                                _total_eval_items = (
                                    len(_chat_samples) + len(_plain_texts)
                                )

                                _sq_accuracy = 0.0
                                _base_loss = float("inf")
                                _think_loss = float("inf")
                                _plain_loss = float("inf")
                                _combined_loss_sum = 0.0
                                _combined_count = 0


                                try:
                                    with _subtensor_lock:
                                        _eval_block = subtensor.get_current_block()
                                        _eval_block_hash = subtensor.get_block_hash(
                                            _eval_block
                                        )
                                except Exception as _bh_err:


                                    import hashlib as _hl
                                    _eval_block_hash = _hl.sha256(
                                        f"{current_epoch_seed}:{_time.time()}".encode()
                                    ).hexdigest()
                                    logging.warning(
                                        f"Block hash fetch failed ({_bh_err}); "
                                        f"using fallback entropy"
                                    )

                                with Progress(
                                    TextColumn("    [cyan]Loss eval[/cyan]"),
                                    BarColumn(bar_width=28),
                                    MofNCompleteColumn(),
                                    TaskProgressColumn(),
                                    TimeElapsedColumn(),
                                    console=console,
                                    transient=False,
                                ) as _prog:
                                    _task = _prog.add_task("", total=_total_eval_items)
                                    _n_chat = len(_chat_samples)

                                    if _chat_samples:
                                        # Single 3-turn conversation per sample
                                        # (2 side quests + 1 real, shuffled).
                                        # Returns think_ce, base_ce, sq_accuracy.
                                        # Mamba2's O(1) recurrent decode state lets us afford a
                                        # much longer thinking trace on a 24 GB GPU than the
                                        # Transformer track (whose KV cache grows linearly with L).
                                        _think_budget = (
                                            EVAL_THINK_MAX_NEW_TOKENS_MAMBA2
                                            if eval_track == "mamba2"
                                            else EVAL_THINK_MAX_NEW_TOKENS
                                        )
                                        _think_loss, _base_loss, _sq_accuracy = evaluate_with_side_quests(
                                            _model_obj, _ref_tokenizer,
                                            _chat_samples,
                                            block_hash=_eval_block_hash,
                                            max_new_tokens=SIDE_QUEST_MAX_NEW_TOKENS,
                                            think_max_new_tokens=_think_budget,
                                            max_length=min(_eval_max_seq, SIDE_QUEST_MAX_CTX),
                                            device=_device,
                                            progress_callback=lambda d, t: (
                                                _prog.update(_task, completed=d)
                                            ),
                                            penalty_loss=EVAL_PENALTY_LOSS,
                                        )
                                        if _think_loss != float("inf"):
                                            _combined_loss_sum += _think_loss * _n_chat
                                            _combined_count += _n_chat

                                    if _plain_texts:
                                        _plain_loss = compute_cross_entropy_loss(
                                            _model_obj, _ref_tokenizer,
                                            _plain_texts,
                                            batch_size=_eval_batch,
                                            max_length=_eval_max_seq,
                                            device=_device,
                                            progress_callback=lambda d, t: (
                                                _prog.update(
                                                    _task,
                                                    completed=_n_chat + d,
                                                )
                                            ),
                                        )
                                        if _plain_loss != float("inf"):
                                            _combined_loss_sum += (
                                                _plain_loss * len(_plain_texts)
                                            )
                                            _combined_count += len(_plain_texts)


                                    _raw_ce_loss = (
                                        _combined_loss_sum / _combined_count
                                        if _combined_count > 0
                                        else float("inf")
                                    )


                                    _loss = _raw_ce_loss

                                    _prog.update(_task, completed=_total_eval_items)

                                _eval_elapsed = _time.monotonic() - _eval_start


                                if _loss == float("inf"):
                                    _loss = EVAL_PENALTY_LOSS


                                progress_tracker.record(
                                    uid=uid,
                                    epoch=epoch_num,
                                    loss=_loss,
                                    thinking_loss=(
                                        _think_loss
                                        if _think_loss != float("inf")
                                        else 0.0
                                    ),
                                    model_revision=revision,
                                    validator_uid=my_uid,
                                    dataset_names=list(datasets_for_eval.keys()),
                                    base_loss=(
                                        _base_loss
                                        if _base_loss != float("inf")
                                        else _loss
                                    ),
                                    sq_accuracy=_sq_accuracy,
                                )

                                _score = progress_tracker.compute_score(uid)
                                _think_disp = (
                                    f"{_think_loss:.4f}"
                                    if _think_loss != float("inf")
                                    else "N/A"
                                )
                                console.print(
                                    f"    Loss [bold]{_loss:.4f}[/bold] | "
                                    f"Think {_think_disp} | "
                                    f"SideQ {_sq_accuracy:.0%} | "
                                    f"Score [bold]{_score:.4f}[/bold] "
                                    f"({_eval_elapsed:.1f}s)"
                                )

                                round_results.append({
                                    'miner_uid': uid,
                                    'miner_hotkey': hotkey,
                                    'track': eval_track,
                                    'model_name': model_name,
                                    'revision': revision,
                                    'loss': _loss,
                                    'thinking_loss': (
                                        _think_loss
                                        if _think_loss != float("inf")
                                        else None
                                    ),
                                    'side_quest_accuracy': _sq_accuracy,
                                    'score': _score,
                                    'datasets': {
                                        n: len(idx)
                                        for n, idx in datasets_for_eval.items()
                                    },
                                    'epoch': epoch_num,
                                    'timestamp': epoch_start.isoformat(),
                                })

                                if use_wandb and wandb_run:
                                    wandb_run.log({
                                        f"{eval_track}/uid_{uid}_loss": _loss,
                                        f"{eval_track}/uid_{uid}_score": _score,
                                        f"{eval_track}/uid_{uid}_sq_accuracy": _sq_accuracy,
                                        f"{eval_track}/uid_{uid}_think_loss": (
                                            _think_loss if _think_loss != float("inf") else None
                                        ),
                                        "epoch": epoch_num,
                                    })

                            except Exception as _eval_err:
                                import traceback as _tb
                                first_line = (
                                    str(_eval_err).splitlines()[0]
                                    if str(_eval_err)
                                    else repr(_eval_err)
                                )
                                console.print(f"    [red]✗ Failed: {first_line}[/red]")
                                console.print(
                                    f"      [dim]{_tb.format_exc().splitlines()[-1]}[/dim]"
                                )


                                progress_tracker.record(
                                    uid=uid,
                                    epoch=epoch_num,
                                    loss=EVAL_PENALTY_LOSS,
                                    thinking_loss=0.0,
                                    model_revision=revision,
                                    validator_uid=my_uid,
                                    dataset_names=list(
                                        datasets_for_eval.keys()
                                    ),
                                    sq_accuracy=0.0,
                                )
                                console.print(
                                    f"    [dim]Recorded penalty loss "
                                    f"({EVAL_PENALTY_LOSS})[/dim]"
                                )

                                skipped.append({
                                    'uid': uid,
                                    'reason': f'error: {first_line[:60]}',
                                })
                                _store_in_cache = False
                                if use_wandb and wandb_run:
                                    wandb_run.log({f"{eval_track}/uid_{uid}_error": 1})

                            finally:
                                import gc as _gc_cleanup
                                import torch as _torch_cleanup

                                if _store_in_cache and not _using_cache:
                                    _model_cache["key"] = cache_key
                                    _model_cache["model"] = _model_obj
                                    _model_cache["tokenizer"] = _tok
                                    _model_cache["cleanup_fn"] = cleanup_fn
                                    cleanup_fn = None
                                    _model_obj = _tok = None
                                    _gc_cleanup.collect()

                                elif not _store_in_cache:
                                    if _model_cache["key"] == cache_key:
                                        _evict_model_cache()
                                    _model_obj = _tok = None
                                    model_validator = None
                                    if cleanup_fn is not None:
                                        try:
                                            cleanup_fn()
                                        except Exception:
                                            pass
                                        cleanup_fn = None
                                    _gc_cleanup.collect()
                                    _torch_cleanup.cuda.empty_cache()
                                    _gc_cleanup.collect()
                                    try:
                                        from evolai.validator.evaluator import (
                                            purge_hf_model_cache,
                                        )
                                        purge_hf_model_cache(model_name)
                                    except Exception:
                                        pass
                                else:
                                    _model_obj = _tok = None
                                    _gc_cleanup.collect()


                        if round_results:
                            results_dir = Path.home() / ".evolai" / "validator" / "results"
                            results_dir.mkdir(parents=True, exist_ok=True)
                            results_file = results_dir / (
                                f"epoch_{epoch_num}_{eval_track}_"
                                f"{epoch_start.strftime('%Y%m%d_%H%M%S')}.json"
                            )
                            with open(results_file, 'w') as _f:
                                json.dump(
                                    {
                                        'track': eval_track,
                                        'epoch': epoch_num,
                                        'mode': 'epoch-based-decentralised',
                                        'seed': current_epoch_seed,
                                        'timestamp': epoch_start.isoformat(),
                                        'results': round_results,
                                        'skipped': skipped,
                                    },
                                    _f,
                                    indent=2,
                                )
                            console.print(
                                f"\n  [green]✓ {eval_track.upper()}:[/green] "
                                f"{len(round_results)} evaluated, "
                                f"{len(skipped)} skipped — {results_file.name}"
                            )
                            if use_wandb and wandb_run:
                                wandb_run.log({
                                    f"{eval_track}/miners_evaluated": len(round_results),
                                    f"{eval_track}/miners_skipped": len(skipped),
                                })


                            try:
                                _owner_results = [
                                    {
                                        'miner_uid': r['miner_uid'],
                                        'miner_hotkey': r['miner_hotkey'],
                                        'raw_score': r['score'],
                                        'effective_score': r['score'],
                                        'dataset_distribution': {
                                            'datasets': r['datasets'],
                                            'track': r['track'],
                                            'model_name': r['model_name'],
                                            'revision': r['revision'],
                                            'loss': r['loss'],
                                            'epoch': r['epoch'],
                                        },
                                    }
                                    for r in round_results
                                ]
                                _submitted = submit_evaluations(
                                    evaluation_round=epoch_num,
                                    judge_model=f"epoch-loss-hf:{eval_track}",
                                    results=_owner_results,
                                    owner_api_url=owner_api,
                                    auth=validator_auth,
                                )
                                if _submitted:
                                    console.print(
                                        f"  [dim]✓ Telemetry sent "
                                        f"({len(_owner_results)} records)[/dim]"
                                    )
                            except Exception as _te_err:
                                logging.warning(
                                    f"[telemetry] submit_evaluations: {_te_err}"
                                )
                        else:
                            console.print(
                                f"  [yellow]No {eval_track} results this epoch.[/yellow]"
                            )

                    except Exception as _track_err:
                        console.print(
                            f"  [red]✗ Track {eval_track} failed: {_track_err}[/red]\n"
                        )
                        if use_wandb and wandb_run:
                            wandb_run.log({f"{eval_track}/track_error": 1})


                _frontier_improved = progress_tracker.update_global_best()
                if _frontier_improved:
                    console.print(
                        f"  [green]✓ Global best EMA loss improved: "
                        f"{progress_tracker.get_best_ema_loss():.6f}[/green]"
                    )

                console.print("[bold]Current Leaderboard:[/bold]\n")
                _leaderboards = _get_track_score_maps(
                    min_evaluations=1, subtensor_lock=_subtensor_lock
                )
                for _tname in active_tracks:
                    _lb_scores = _leaderboards[_tname]
                    if not _lb_scores:
                        console.print(
                            f"  [dim]{_tname.upper()}: no miners scored yet.[/dim]"
                        )
                        continue
                    console.print(f"[bold]{_tname.upper()}[/bold]")
                    _sorted = sorted(
                        _lb_scores.items(), key=lambda x: x[1], reverse=True
                    )
                    _lb_table = Table(show_header=True, header_style="bold cyan")
                    _lb_table.add_column("Rank", justify="right", style="dim")
                    _lb_table.add_column("UID", justify="right")
                    _lb_table.add_column("Score", justify="right", style="bold")
                    _lb_table.add_column("Latest Loss", justify="right")
                    _lb_table.add_column("Evals", justify="right")
                    for _rank, (_uid_lb, _sc) in enumerate(_sorted[:15], 1):
                        _st = progress_tracker.get_miner_state(_uid_lb)
                        _ll = progress_tracker.get_latest_loss(_uid_lb)
                        _ll_str = f"{_ll:.4f}" if _ll is not None else "N/A"
                        _ec = str(_st.eval_count) if _st else "0"
                        _lb_table.add_row(
                            str(_rank), str(_uid_lb), f"{_sc:.4f}", _ll_str, _ec
                        )
                    console.print(_lb_table)
                    console.print()


                try:
                    with _subtensor_lock:
                        _cur_block = subtensor.get_current_block()
                    _remaining = EPOCH_BLOCKS - (_cur_block % EPOCH_BLOCKS)
                except Exception:
                    _remaining = 60

                _epoch_sleep_s = max(30, _remaining * 12)
                _elapsed_s = (datetime.utcnow() - epoch_start).total_seconds()
                _min_round_s = 3600
                _sleep_s = max(_min_round_s - _elapsed_s, _epoch_sleep_s)
                _sleep_s = max(30, _sleep_s)
                console.print(
                    f"[dim]Epoch {epoch_num} complete ({_elapsed_s:.0f}s elapsed). "
                    f"Sleeping {_sleep_s // 60:.0f}m {_sleep_s % 60:.0f}s "
                    f"(min 1h enforced)…[/dim]\n"
                )
                console.print("━" * 80 + "\n")
                _stop_requested.wait(timeout=_sleep_s)

        except KeyboardInterrupt:
            _stop_requested.set()

        except Exception as _crash_err:
            import traceback as _tb
            crash_count += 1
            logging.error(
                f"[run] Crash #{crash_count}: {_crash_err}", exc_info=True
            )
            console.print(
                f"\n[red]Crash #{crash_count}/{MAX_CRASH_RESTARTS}: "
                f"{_crash_err}[/red]"
            )
            console.print(f"[dim]{_tb.format_exc().splitlines()[-1]}[/dim]")
            _evict_model_cache()

            if crash_count >= MAX_CRASH_RESTARTS:
                console.print(
                    f"[red]Exceeded {MAX_CRASH_RESTARTS} crashes — exiting.[/red]"
                )
                _stop_requested.set()
                break

            backoff = min(300, CRASH_BACKOFF_BASE_S * crash_count)
            console.print(
                f"[yellow]Restarting in {backoff}s (crash #{crash_count})…[/yellow]\n"
            )
            if use_wandb and wandb_run:
                try:
                    wandb_run.log({
                        "validator_crash": 1, "crash_count": crash_count
                    })
                except Exception:
                    pass
            if _stop_requested.wait(timeout=backoff):
                break
            crash_count = max(0, crash_count - 1)
            continue


    console.print("\n[yellow]Validator stopped[/yellow]")
    _evict_model_cache()

    if use_wandb and wandb_run:
        wandb_run.finish()

    total_runtime_h = (datetime.utcnow() - _run_start).total_seconds() / 3600
    console.print(f"\n[bold]Session Summary:[/bold]")
    console.print(f"  Total Epochs: {epoch_count}")
    console.print(f"  Total Runtime: {total_runtime_h:.2f} hours")
    console.print()


if __name__ == "__main__":
    validator_app()
