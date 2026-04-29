"""Setup configuration for EvolAI Package"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = [
    "typer>=0.12.0",
    "rich>=13.7.0", 
    "httpx>=0.27.0",
    "bittensor-wallet>=4.0.0",
    "bittensor>=9.4.0",
    "transformers>=5.3.0",
    "torch>=2.0.0",
    "accelerate>=0.20.0",
    "huggingface_hub>=0.20.0",
    "wandb>=0.16.0",
    "python-dotenv>=1.0.0",
    "datasets>=2.14.0",
    "psutil>=5.9.0",
]

setup(
    name="evolai",
    version="0.2.0",
    description="EvolAI - LLM Model Evaluation Subnet on Bittensor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="EvolAI Team",
    url="https://github.com/evolai-subnet/evolai",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "docs*"]),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "evolcli=evolai.cli.main:cli_main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="bittensor subnet llm evaluation blockchain",
)
