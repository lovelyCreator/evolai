
import typer
import json
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
import os
from pathlib import Path
console = Console()
err_console = Console(stderr=True, style="bold red")

miner_app = typer.Typer(help="Commands for miners to register and validate models")


def check_model_eligibility(model_name: str, revision: str = "main", track: str = "transformer") -> dict:
    issues = []
    warnings = []
    info = {}
    

    model_name_lower = model_name.lower()

    if "/" in model_name:
        repo_name, model_only = model_name.split("/", 1)
        if "evolai" not in model_only.lower():
            issues.append("Model name must include 'evolai' (e.g., username/evolai-my-model-9b)")
    else:
        issues.append("Model name must be in format: username/evolai-model-name")
    
    console.print(f"\n[cyan]Loading model from HuggingFace: {model_name} (revision: {revision})[/cyan]")
    
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        import torch
    except ImportError as e:
        missing_package = str(e).split("'")[1] if "'" in str(e) else "unknown"
        issues.append(
            f"Required package '{missing_package}' not installed. "
            f"Install with: pip install transformers torch"
        )
        return {"eligible": False, "issues": issues, "warnings": warnings, "info": info}
    
    try:
        

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task1 = progress.add_task("Loading model config...", total=None)


            config = AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=False)
            info["revision"] = revision or "latest"
            progress.update(task1, completed=True)
            

            task2 = progress.add_task("Calculating model size...", total=None)
            

            console.print("[cyan]Downloading model to verify eligibility (this may take a while)...[/cyan]")


            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                trust_remote_code=False,
                torch_dtype=torch.bfloat16,
                device_map="cpu"
            )
            num_params = sum(p.numel() for p in model.parameters())
            num_params_b = num_params / 1e9
            info["num_parameters"] = num_params
            info["num_parameters_b"] = num_params_b
            

            from evolai.validator.config import VALID_PARAM_RANGES_B
            matched_range = next(
                ((lo, hi) for lo, hi in VALID_PARAM_RANGES_B if lo <= num_params_b <= hi), None
            )
            
            if matched_range is None:
                ranges_str = ", ".join(f"{lo:.2f}-{hi:.2f}B" for lo, hi in VALID_PARAM_RANGES_B)
                issues.append(
                    f"Model has {num_params_b:.2f}B parameters. Must be one of: {ranges_str}"
                )
            else:
                lo, hi = matched_range
                size_category = f"{lo:.2f}-{hi:.2f}B"
                console.print(f"[green]✓[/green] Model size: {num_params_b:.2f}B (within {size_category} range)")
                info["size_category"] = size_category
            
            del model
            torch.cuda.empty_cache()
            progress.update(task2, completed=True)
            

            task3 = progress.add_task("Checking architecture...", total=None)
            

            architecture = config.model_type if hasattr(config, 'model_type') else "unknown"
            info["architecture"] = architecture
            

            if track == "mamba2":

                if "mamba" not in architecture.lower() and "recurrent" not in architecture.lower():
                    warnings.append(
                        "Model architecture does not appear to be Mamba2/recurrent. "
                        "Ensure your model uses Mamba2 or recurrent mechanisms."
                    )
                console.print(f"[cyan]Architecture: {architecture} (Mamba2 Track)[/cyan]")
                info["vllm_compatible"] = False
                
            elif track == "transformer":


                vllm_architectures = [
                    "afmoe", "apertus", "aquila", "arcee", "arctic", "baichuan", "bailing_moe",
                    "bamba", "bloom", "chatglm", "cohere", "cohere2", "dbrx", "decilm",
                    "deepseek", "deepseek_v2", "deepseek_v3", "dots1", "dots_ocr",
                    "ernie4_5", "ernie4_5_moe", "exaone", "exaone4", "exaone_moe",
                    "fairseq2_llama", "falcon", "falcon_h1", "falcon_mamba", "flex_olmo",
                    "gemma", "gemma2", "gemma3", "gemma3n",
                    "glm", "glm4", "glm4_moe", "gpt2", "gpt_bigcode", "gpt_j", "gptj", "gpt_neox", "gpt_oss",
                    "granite", "granitemoe", "granitemoehybrid", "granitemoeshared", "gritlm", "grok1",
                    "hunyuan", "hunyuan_v1", "internlm", "internlm2", "internlm3",
                    "jais", "jais2", "jamba", "kimi_linear",
                    "lfm2", "lfm2_moe", "llama", "llama4",
                    "mamba", "mamba2", "mimo", "minicpm", "minicpm3",
                    "minimax_m2", "minimax_text", "mistral", "mistral_large_3", "mixtral", "mpt",
                    "nemotron", "nemotron_h", "olmo", "olmo2", "olmo3", "olmoe", "opt", "orion", "ouro",
                    "persimmon", "phi", "phi3", "phimoe", "plamo2", "plamo3",
                    "qwen", "qwen2", "qwen2_moe", "qwen3", "qwen3_moe", "qwen3_next",
                    "seed_oss", "solar", "stablelm", "starcoder2", "step1",
                    "telechat2", "teleflm", "xverse", "yi", "zamba2", "longcat_flash"
                ]
                is_vllm_compatible = any(arch in architecture.lower() for arch in vllm_architectures)
                
                if is_vllm_compatible:
                    console.print(f"[green]✓[/green] Architecture: {architecture} (vLLM-compatible)")
                    info["vllm_compatible"] = True
                else:
                    warnings.append(
                        f"Architecture '{architecture}' may not be vLLM-compatible. "
                        "Validators will fall back to HuggingFace transformers."
                    )
                    console.print(f"[yellow]⚠[/yellow] Architecture: {architecture} (HuggingFace transformers only)")
                    info["vllm_compatible"] = False
            
            progress.update(task3, completed=True)
            

            task4 = progress.add_task("Checking HuggingFace interface...", total=None)
            
            required_attrs = ["hidden_size", "num_hidden_layers", "num_attention_heads"]
            missing_attrs = [attr for attr in required_attrs if not hasattr(config, attr)]
            
            if missing_attrs:
                warnings.append(
                    f"Config missing standard attributes: {', '.join(missing_attrs)}. "
                    "Ensure your model implements HuggingFace interfaces."
                )
            else:
                console.print("[green]✓[/green] HuggingFace interface: Compatible")
            
            progress.update(task4, completed=True)
            
    except Exception as e:
        issues.append(f"Failed to load model: {str(e)}")
        return {"eligible": False, "issues": issues, "warnings": warnings, "info": info}
    

    eligible = len(issues) == 0
    
    return {
        "eligible": eligible,
        "issues": issues,
        "warnings": warnings,
        "info": info
    }


@miner_app.command("check")
def check_model(
    model_name: Optional[str] = typer.Argument(
        None,
        help="HuggingFace model name (e.g., evolai/my-model-9b)"
    ),
    revision: Optional[str] = typer.Option(
        None,
        "--revision",
        "-r",
        help="Git revision (branch, tag, or commit SHA)"
    ),
    track: Optional[str] = typer.Option(
        None,
        "--track",
        "-t",
        help="Competition track: 'transformer' or 'mamba2'"
    )
):

    if model_name is None:
        console.print("\n[bold cyan]EvolAI Model Eligibility Check[/bold cyan]")
        console.print("─" * 60)
        model_name = Prompt.ask(
            "[cyan]Enter HuggingFace model name[/cyan]",
            default="myorg/evolai-model-9b"
        )
    
    if track is None:
        console.print("\n[yellow]Available tracks:[/yellow]")
        console.print("  1. transformer - Standard transformer models (50% emissions)")
        console.print("  2. mamba2 - Recurrent/stateful models (50% emissions)")
        track_choice = Prompt.ask(
            "[cyan]Select track[/cyan]",
            choices=["transformer", "mamba2", "1", "2"],
            default="transformer"
        )

        if track_choice == "1":
            track = "transformer"
        elif track_choice == "2":
            track = "mamba2"
        else:
            track = track_choice

    if revision is None:
        revision_input = Prompt.ask(
            "[cyan]Enter model revision (press Enter for latest)[/cyan]",
            default=""
        )

        revision = revision_input.strip() if revision_input.strip() else None
    
    if track not in ["transformer", "mamba2"]:
        err_console.print("Invalid track. Use 'transformer' or 'mamba2'.")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold cyan]Model Eligibility Check[/bold cyan]\n"
        f"Model: {model_name}\n"
        f"Revision: {revision or 'latest'}\n"
        f"Track: {track.upper()}",
        border_style="cyan"
    ))
    
    result = check_model_eligibility(model_name, revision, track)
    

    console.print("\n" + "=" * 60)
    
    if result["eligible"]:
        console.print("\n[bold green]✓ MODEL IS ELIGIBLE[/bold green]\n")
    else:
        console.print("\n[bold red]✗ MODEL IS NOT ELIGIBLE[/bold red]\n")
    

    if result["issues"]:
        console.print("[bold red]Issues (must fix):[/bold red]")
        for issue in result["issues"]:
            console.print(f"  [red]✗[/red] {issue}")
        console.print()
    

    if result["warnings"]:
        console.print("[bold yellow]Warnings (check carefully):[/bold yellow]")
        for warning in result["warnings"]:
            console.print(f"  [yellow]⚠[/yellow] {warning}")
        console.print()
    

    if result["info"]:
        console.print("[bold cyan]Model Information:[/bold cyan]")
        table = Table(show_header=False, box=None)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="bold")
        
        info = result["info"]
        if "revision" in info:
            table.add_row("Revision", info["revision"])
        if "vocab_size" in info:
            table.add_row("Tokenizer Vocab Size", str(info["vocab_size"]))
        if "num_parameters_b" in info:
            table.add_row("Model Size", f"{info['num_parameters_b']:.2f}B parameters")
        if "size_category" in info:
            table.add_row("Size Category", info["size_category"])
        if "architecture" in info:
            table.add_row("Architecture", info["architecture"])
        if "vllm_compatible" in info:
            vllm_status = "Yes" if info["vllm_compatible"] else "No"
            table.add_row("vLLM Compatible", vllm_status)
        
        console.print(table)
    
    console.print("\n" + "=" * 60)
    
    if not result["eligible"]:
        raise typer.Exit(1)


@miner_app.command("register")
def register_model(
    model_name: Optional[str] = typer.Argument(
        None,
        help="HuggingFace model name (e.g., evolai/my-model-9b)"
    ),
    wallet_name: Optional[str] = typer.Option(
        None,
        "--wallet-name",
        "--wallet",
        "-w",
        help="Name of the wallet (hotkey)"
    ),
    wallet_path: str = typer.Option(
        "~/.bittensor/wallets",
        "--wallet-path",
        help="Path to wallet directory"
    ),
    hotkey: Optional[str] = typer.Option(
        None,
        "--hotkey",
        help="Hotkey name"
    ),
    revision: Optional[str] = typer.Option(
        None,
        "--revision",
        "-r",
        help="Git revision (branch, tag, or commit SHA)"
    ),
    track: Optional[str] = typer.Option(
        None,
        "--track",
        "-t",
        help="Competition track: 'transformer' or 'mamba2'"
    ),
    netuid: int = typer.Option(
        47,
        "--netuid",
        help="Subnet netuid"
    )
):

    console.print("\n[bold cyan]EvolAI Model Registration[/bold cyan]")
    console.print("━" * 60)
    

    if track is None:
        console.print("\n[yellow]Available competition tracks (you can only register for ONE):[/yellow]")
        console.print("  1. transformer - Standard transformer models (50% emissions)")
        console.print("  2. mamba2 - Recurrent/stateful models (50% emissions)")
        track_choice = Prompt.ask(
            "[cyan]Select track[/cyan]",
            choices=["transformer", "mamba2", "1", "2"],
            default="transformer"
        )

        if track_choice == "1":
            track = "transformer"
        elif track_choice == "2":
            track = "mamba2"
        else:
            track = track_choice
    

    if track not in ["transformer", "mamba2"]:
        err_console.print("Invalid track. Use 'transformer' or 'mamba2'.")
        raise typer.Exit(1)
    

    if model_name is None:
        default_name = "myorg/evolai-mamba2-9b" if track == "mamba2" else "myorg/evolai-transformer-9b"
        model_name = Prompt.ask(
            f"[cyan]Enter {track} model name[/cyan]",
            default=default_name
        )
    
    if revision is None:
        revision = Prompt.ask(
            "[cyan]Enter model revision[/cyan]",
            default="main"
        ).strip() or "main"
    

    if wallet_name is None:
        wallet_name = Prompt.ask(
            "[cyan]Enter wallet name[/cyan]",
            default="default"
        )
    

    if hotkey is None or hotkey == "default":
        wallet_dir = Path(wallet_path).expanduser() / wallet_name / "hotkeys"
        if wallet_dir.exists():
            available_hotkeys = [f.stem for f in wallet_dir.iterdir() if f.is_file()]
            if available_hotkeys:
                console.print(f"\n[yellow]Available hotkeys for wallet '{wallet_name}':[/yellow]")
                for i, hk in enumerate(available_hotkeys, 1):
                    console.print(f"  {i}. {hk}")
                
                hotkey_choice = Prompt.ask(
                    "[cyan]Select hotkey[/cyan]",
                    choices=available_hotkeys + [str(i) for i in range(1, len(available_hotkeys) + 1)],
                    default=available_hotkeys[0] if available_hotkeys else "default"
                )
                

                if hotkey_choice.isdigit():
                    idx = int(hotkey_choice) - 1
                    if 0 <= idx < len(available_hotkeys):
                        hotkey = available_hotkeys[idx]
                else:
                    hotkey = hotkey_choice
            else:
                console.print(f"\n[yellow]⚠ No hotkeys found for wallet '{wallet_name}' at {wallet_dir}[/yellow]")
                hotkey = Prompt.ask("[cyan]Enter hotkey name[/cyan]", default="default")
        else:
            console.print(f"\n[yellow]⚠ Wallet directory not found: {wallet_dir}[/yellow]")
            hotkey = Prompt.ask("[cyan]Enter hotkey name[/cyan]", default="default")
    
    
    _register_single_track(
        model_name, revision, track,
        wallet_name, wallet_path, hotkey, netuid
    )


def _register_single_track(
    model_name: str,
    revision: str,
    track: str,
    wallet_name: str,
    wallet_path: str,
    hotkey: str,
    netuid: int
):
    
    console.print(Panel.fit(
        f"[bold cyan]Model Registration[/bold cyan]\n"
        f"Model: {model_name}\n"
        f"Revision: {revision}\n"
        f"Track: {track.upper()}\n"
        f"Wallet: {wallet_name}\n"
        f"Hotkey: {hotkey}",
        border_style="cyan"
    ))
    

    console.print("\n[cyan]Registering model on-chain...[/cyan]")
    
    try:
        from bittensor_wallet import Wallet
        import bittensor as bt
        from evolai.utils.metadata import compress_metadata
        

        wallet = Wallet(name=wallet_name, path=wallet_path, hotkey=hotkey)
        if not wallet.hotkey_file.exists_on_device():
            err_console.print(f"Hotkey not found for wallet '{wallet_name}' at {wallet_path}")
            raise typer.Exit(1)
        
        hotkey_address = wallet.hotkey.ss58_address
        console.print(f"[green]✓[/green] Wallet loaded: [bold]{hotkey_address}[/bold]")
        

        subtensor = bt.Subtensor(network="finney")
        console.print("[green]✓[/green] Connected to Bittensor network")
        

        metagraph = subtensor.metagraph(netuid)
        if hotkey_address not in metagraph.hotkeys:
            err_console.print(f"Hotkey {hotkey_address} is not registered in subnet {netuid}")
            err_console.print("Please register first using: btcli subnet register")
            raise typer.Exit(1)
        
        uid = metagraph.hotkeys.index(hotkey_address)
        console.print(f"[green]✓[/green] Found UID: {uid}")
        

        console.print(f"\n[cyan]Committing model metadata to chain...[/cyan]")
        console.print(f"  Model: {model_name}")
        console.print(f"  Revision: {revision or 'main'}")
        console.print(f"  Track: {track}")
        console.print(f"  [dim]Note: Validators will verify ownership from HuggingFace when evaluating[/dim]")
        

        metadata = {
            track: {
                "model_name": model_name,
                "revision": revision or "main"
            }
        }
        

        compressed_metadata = compress_metadata(metadata)
        

        if len(compressed_metadata) > 128:
            err_console.print(f"\n[bold red]✗ Metadata too large: {len(compressed_metadata)} bytes (max 128)[/bold red]")
            err_console.print(f"[yellow]Model name too long. Try shorter HuggingFace username or model name.[/yellow]")
            raise typer.Exit(1)
        
        console.print(f"  [dim]Metadata size: {len(compressed_metadata)} bytes[/dim]")
        

        metadata_dir = Path.home() / ".evolai" / "miner" / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = metadata_dir / f"uid_{uid}_{track}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        

        response = subtensor.set_commitment(
            wallet=wallet,
            netuid=netuid,
            data=compressed_metadata
        )
        
        success = response if isinstance(response, bool) else getattr(response, "success", False)
        if success:
            console.print("\n[bold green]✓ Model registered successfully![/bold green]")
            console.print(f"[dim]UID: {uid} | Model: {model_name}@{revision} | Track: {track}[/dim]")
            
            console.print("\n[cyan]Next steps:[/cyan]")
            console.print("1. Ensure your model is publicly accessible on HuggingFace")
            console.print("2. Wait for validators to evaluate your model")
            console.print("3. Monitor your score on the subnet dashboard")
            console.print("\n[yellow]Note: Each miner can only register for ONE track.[/yellow]")
            console.print("[yellow]To change tracks, re-register with a different --track option.[/yellow]")
        else:
            message = getattr(response, "message", str(response))
            err_console.print(f"Failed to register model on-chain: {message}")
            raise typer.Exit(1)
            
    except Exception as e:
        err_console.print(f"Registration failed: {str(e)}")
        raise typer.Exit(1)


@miner_app.command("info")
def show_info():
    info_text = """
[bold cyan]EvolAI Subnet 47 - Miner Information[/bold cyan]

[bold yellow]Competition Tracks:[/bold yellow]

  [bold]1. Transformer Track (50% emissions)[/bold]
     • Standard transformer architectures (Llama, Qwen, Mistral, etc.)
     • Evaluated via vLLM server for fast inference
     • Valid sizes: ~450M, ~1.6B, ~3.7B, ~9B, ~21B
     
  [bold]2. Mamba2 Track (50% emissions)[/bold]
     • Mamba2/recurrent architectures (selective state space models)
     • Evaluated with HuggingFace transformers
     • Valid sizes: ~450M, ~1.6B, ~3.7B, ~9B, ~21B

  [dim]Note: Each track receives 50% of total subnet emissions.
  [bold yellow]Each miner can only register for ONE track at a time.[/bold yellow]
  To change tracks, re-register with a different track option.[/dim]

[bold yellow]Requirements (Both Tracks):[/bold yellow]

  • Model name: Must include [cyan]evolai[/cyan] in the model part (after /)
    Examples: myorg/evolai-transformer-9b, username/evolai-mamba2-3b
  • Size: one of the allowed parameter ranges:
      0.45–0.48B (~450M) | 1.5–1.8B | 3.5–3.8B | 9.0–9.5B | 21.0–21.5B
  • Interface: Must implement HuggingFace interfaces
    - model.generate() method
    - forward() with input_ids and attention_mask
    - PretrainedConfig class
  
[bold yellow]Evaluation:[/bold yellow]

    Validators evaluate registered models on scheduled challenges.
    Rewards and on-chain weights are based on model performance over time.
  
[bold yellow]Useful Commands:[/bold yellow]

  • Check eligibility:    [cyan]evolcli miner check myorg/evolai-model-9b[/cyan]
  • Register model:       [cyan]evolcli miner register myorg/evolai-model-9b -w miner1[/cyan]
  • Register transformer: [cyan]evolcli miner register myorg/evolai-transformer-9b -w miner1 --track transformer[/cyan]
  • Register Mamba2:      [cyan]evolcli miner register myorg/evolai-mamba2-9b -w miner1 --track mamba2[/cyan]
  • View subnet info:     [cyan]evolcli info[/cyan]

[bold yellow]Tips:[/bold yellow]

  • Always run [cyan]evolcli miner check[/cyan] before registering
  • Ensure your model is public on HuggingFace
  • You can only be registered on ONE track at a time
  • To switch tracks, re-register with a different --track option
    • Keep your model updated and re-register after publishing a new version
    """
    
    console.print(Panel(info_text, border_style="cyan", padding=(1, 2)))


@miner_app.command("get-challenge")
def get_challenge(
    uid: int = typer.Argument(..., help="Your numeric UID on the subnet (integer, e.g. 42)"),
    network: str = typer.Option(
        "finney",
        "--network",
        help="Bittensor network (finney, test, local)",
    ),
    netuid: int = typer.Option(
        47,
        "--netuid",
        help="Subnet netuid",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write challenge to this .txt file (default: challenge_uid<UID>.txt in current directory)",
    ),
):
    from datetime import datetime as _dt

    if not (0 <= uid <= 255):
        err_console.print("UID must be between 0 and 255")
        raise typer.Exit(1)


    console.print(f"[cyan]Connecting to {network}...[/cyan]")
    try:
        import bittensor as bt
        from evolai.validator.epoch_manager import (
            current_epoch,
            derive_indices,
            read_all_validator_seeds,
            EVAL_SALT,
        )
        from evolai.validator.config import (
            ACTIVE_DATASETS,
            DATASET_SIZES,
            EPOCH_BLOCKS,
            N_EVAL,
        )
    except ImportError as e:
        err_console.print(
            f"Missing dependency: {e}\n"
            "Install with: pip install bittensor evolai"
        )
        raise typer.Exit(1)

    try:
        subtensor = bt.Subtensor(network=network)
        metagraph = subtensor.metagraph(netuid=netuid)
        current_block = subtensor.block
        epoch_num = current_epoch(current_block, EPOCH_BLOCKS)
    except Exception as e:
        err_console.print(f"Failed to connect to {network}: {e}")
        raise typer.Exit(1)

    def _close_subtensor() -> None:
        try:
            subtensor.close()
        except Exception:
            pass

    console.print(
        f"[green]✓[/green] Connected — block {current_block}, "
        f"epoch {epoch_num}, {len(metagraph.hotkeys)} UIDs"
    )


    console.print("[cyan]Reading validator seeds from chain...[/cyan]")
    seeds = read_all_validator_seeds(
        subtensor, netuid, metagraph, epoch_num, max_epoch_lag=1
    )

    if not seeds:
        _close_subtensor()
        err_console.print(
            "No validator seeds found on-chain for the current epoch.\n"
            "Validators may not have committed yet. Try again in a few minutes."
        )
        raise typer.Exit(1)

    console.print(
        f"[green]✓[/green] Found {len(seeds)} validator seed(s) "
        f"for epoch ≥ {epoch_num - 1}"
    )

    # For each validator, simulate exactly the challenge they will use for this miner.
    # Each validator uses its own seed → separate, independent test sets.
    per_validator_challenges: list = []
    union_indices: dict = {ds_name: set() for ds_name in ACTIVE_DATASETS}

    for s in seeds:
        v_datasets: dict = {}
        for ds_name in ACTIVE_DATASETS:
            ds_size = DATASET_SIZES.get(ds_name)
            if not ds_size:
                continue
            indices = derive_indices(
                seed=s.seed,
                uid=uid,
                dataset_name=ds_name,
                dataset_size=ds_size,
                n=N_EVAL,
                salt=EVAL_SALT,
            )
            v_datasets[ds_name] = indices
            union_indices[ds_name].update(indices)

        if v_datasets:
            per_validator_challenges.append({
                "validator_uid": s.validator_uid,
                "validator_hotkey": s.validator_hotkey,
                "epoch": s.epoch,
                "seed_prefix": s.seed[:8] + "…",
                "datasets": {
                    ds_name: {"indices": idx, "count": len(idx)}
                    for ds_name, idx in v_datasets.items()
                },
            })

    if not per_validator_challenges:
        _close_subtensor()
        err_console.print("No datasets available for challenge derivation.")
        raise typer.Exit(1)

    # Summary table: one row per validator
    table = Table(
        title=f"Eval Challenge for UID {uid} — {len(per_validator_challenges)} validator(s)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Validator UID", style="bold")
    table.add_column("Hotkey (prefix)")
    table.add_column("Epoch")
    table.add_column("Seed")
    for ds_name in ACTIVE_DATASETS:
        table.add_column(f"{ds_name} (#idx)")

    for vc in per_validator_challenges:
        row = [
            str(vc["validator_uid"]),
            vc["validator_hotkey"][:12] + "…",
            str(vc["epoch"]),
            vc["seed_prefix"],
        ]
        for ds_name in ACTIVE_DATASETS:
            count = vc["datasets"].get(ds_name, {}).get("count", 0)
            row.append(str(count))
        table.add_row(*row)

    console.print(table)

    # Print union summary
    console.print("\n[bold]Union of all validator test sets:[/bold]")
    for ds_name, idx_set in union_indices.items():
        if idx_set:
            console.print(f"  {ds_name}: {len(idx_set)} unique indices")

    console.print(
        f"\n[dim]Train your model to minimise loss on all these texts, "
        f"then register it.[/dim]"
    )

    output_path = Path(output) if output else Path(f"challenge_uid{uid}.json")
    fetched_at = _dt.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    challenge_data = {
        "uid": uid,
        "epoch": epoch_num,
        "block": current_block,
        "network": network,
        "netuid": netuid,
        "fetched_at": fetched_at,
        "validator_count": len(per_validator_challenges),
        "validators": per_validator_challenges,
        "union": {
            ds_name: {
                "indices": sorted(idx_set),
                "count": len(idx_set),
            }
            for ds_name, idx_set in union_indices.items()
            if idx_set
        },
    }

    _close_subtensor()

    try:
        output_path.write_text(
            json.dumps(challenge_data, indent=2), encoding="utf-8"
        )
        console.print(f"[green]✓ Challenge written to:[/green] {output_path.resolve()}")
    except OSError as _write_err:
        err_console.print(f"Failed to write challenge file: {_write_err}")
        raise typer.Exit(1)

    raise typer.Exit(0)
