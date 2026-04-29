# EvolAI

LLM model evaluation subnet on Bittensor.

## Installation

Install [uv](https://github.com/astral-sh/uv), then:

```bash
git clone https://github.com/evolai-subnet/evolai.git
uv pip install -e .
```

Or with pip:

```bash
pip install -e .
```

Verify:

```bash
evolcli --help
```

## Mining

Requirements:

- Model name must contain `evolai`
- Model must be public on HuggingFace
- Supported tracks: `transformer`, `mamba2`

Check eligibility:

```bash
evolcli miner check --model username/evolai-0.4b --track transformer
evolcli miner check --model username/evolai-mamba2-0.4b --track mamba2
```

Get your challenge:

```bash
evolcli miner get-challenge <validator-uid>
```

Register your model:

```bash
evolcli miner register --wallet-name miner1 --hotkey my-hotkey --track transformer
evolcli miner register --wallet-name miner1 --hotkey my-hotkey --track mamba2
```

Re-register after you publish a new model version.

## Validating

Install validator dependencies:

```bash
uv pip install -e ".[validator]"
evolcli validator setup
```

Run the validator:

```bash
evolcli validator run \
  --wallet validator1 \
  --hotkey default
```

A GPU 80 GB is required to run validator evaluations.

Copy [.env.example](.env.example) to `.env` and fill in your credentials before starting.
