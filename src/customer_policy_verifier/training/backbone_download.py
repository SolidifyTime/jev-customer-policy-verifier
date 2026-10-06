"""Download official Qwen files from ModelScope, then load them locally.

Use the current master branch. Only required files and model type are checked;
this project does not pin model commits or compare model-file hashes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SOURCE_FORMAT = "cpv-backbone-source-v3"
TOKENIZER_PATTERNS = ["config.json", "tokenizer*", "vocab.json", "merges.txt", "special_tokens_map.json"]
WEIGHT_PATTERNS = ["*.safetensors", "*.safetensors.index.json", "generation_config.json"]


def sha256(path):
    """Streaming digest for datasets, our checkpoints and upload archives."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(directory, name):
    path = directory / name
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty backbone file: {name}")
    return path


def _has_weights(directory, required=False):
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        try:
            weights = json.loads(index.read_text())["weight_map"]
            if not isinstance(weights, dict) or not weights:
                raise ValueError("empty weight_map")
            for name in set(weights.values()):
                if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
                    raise ValueError("invalid safetensors shard filename")
                _require_file(directory, name)
        except (KeyError, TypeError, ValueError) as error:
            if required:
                raise ValueError(f"incomplete safetensors weights: {error}") from error
            return False
        return True
    weights = directory / "model.safetensors"
    present = weights.is_file() and weights.stat().st_size > 0
    if required and not present:
        raise ValueError("only tokenizer downloaded or weights missing; run download without --tokenizer-only on the GPU host")
    return present


def validate_backbone(path, cfg, require_weights=False):
    directory = Path(path)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        _require_file(directory, name)
    if json.loads((directory / "config.json").read_text()).get("model_type") != "qwen3":
        raise ValueError("expected a Qwen3 backbone configuration")
    source_path = directory / "backbone-source.json"
    source = json.loads(source_path.read_text()) if source_path.exists() else {
        "provider": "local", "repo_id": cfg["model_repo"], "revision": "local"}
    if source.get("repo_id") != cfg["model_repo"]:
        raise ValueError("backbone directory belongs to a different model repository")
    # Old source records remain readable. Revisions and recorded hashes are
    # informational only; native Transformers verifies actual model loading.
    return dict(source, has_weights=_has_weights(directory, required=require_weights))


def source_digest(path):
    """Optional digest of the small source note, never of model weight files."""
    source = Path(path) / "backbone-source.json"
    return sha256(source) if source.is_file() else None


def download(args, cfg):
    if cfg.get("download_source", "modelscope") != "modelscope":
        raise ValueError("the download command uses ModelScope")
    from modelscope.hub.snapshot_download import snapshot_download

    directory = Path(args.backbone)
    directory.mkdir(parents=True, exist_ok=True)
    patterns = TOKENIZER_PATTERNS + ([] if args.tokenizer_only else WEIGHT_PATTERNS)
    revision = cfg.get("model_revision") or "master"
    downloaded = snapshot_download(model_id=cfg["model_repo"], revision=revision,
                                   local_dir=str(directory.resolve()), allow_patterns=patterns)
    if Path(downloaded).resolve() != directory.resolve():
        raise ValueError("ModelScope returned an unexpected download directory")
    result = validate_backbone(directory, cfg, require_weights=not args.tokenizer_only)
    files = sorted(p.name for p in directory.iterdir() if p.is_file()
                   and p.name != "backbone-source.json" and not p.name.startswith("."))
    source = {"format": SOURCE_FORMAT, "provider": "modelscope", "repo_id": cfg["model_repo"],
              "revision": revision, "versioning": "current branch; no project content pins",
              "has_weights": result["has_weights"], "files": files}
    target = directory / "backbone-source.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(target)
    return {"backbone": str(directory.resolve()), "source": source}
