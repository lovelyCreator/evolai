
import json
from datetime import datetime
from typing import Dict, Union


def compress_metadata(metadata: dict) -> str:
    compressed = {}
    

    if "transformer" in metadata:
        t = metadata["transformer"]
        compressed["t"] = {
            "m": t["model_name"],
            "r": t.get("revision") or "main"
        }
    

    if "mamba2" in metadata:
        s = metadata["mamba2"]
        compressed["m2"] = {
            "m": s["model_name"],
            "r": s.get("revision") or "main"
        }
    
    return json.dumps(compressed, separators=(',', ':'))


def decompress_metadata(compressed_str: Union[str, bytes]) -> dict:
    compressed = json.loads(compressed_str)
    metadata = {}
    

    if "t" in compressed:
        t = compressed["t"]
        metadata["transformer"] = {
            "model_name": t["m"],
            "revision": t["r"]
        }
    

    if "m2" in compressed:
        s = compressed["m2"]
        metadata["mamba2"] = {
            "model_name": s["m"],
            "revision": s["r"]
        }
    

    if "s" in compressed:
        s = compressed["s"]
        metadata["mamba2"] = {
            "model_name": s["m"],
            "revision": s["r"]
        }
    
    metadata["version"] = compressed.get("v", "1.0")
    
    return metadata
