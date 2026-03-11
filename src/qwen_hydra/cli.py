"""
CLI entrypoints for Qwen-Hydra.

  qwen-hydra extract --output ./deltas
  qwen-hydra info    --dir ./deltas
"""

import json
import logging
import sys
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-24s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)


@click.group()
def main():
    """Qwen-Hydra: one trunk, three heads."""
    pass


@main.command()
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to write trunk + deltas into.",
)
@click.option(
    "--size", "-s",
    type=click.Choice(["0.6B", "4B", "8B"], case_sensitive=False),
    default="0.6B",
    help="Model size to extract (default: 0.6B).",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="HuggingFace cache directory for model downloads.",
)
@click.option(
    "--threshold",
    type=float,
    default=1e-8,
    help="Sparsity threshold: deltas smaller than this are treated as zero.",
)
def extract(output: Path, size: str, cache_dir: Path, threshold: float):
    """Download models and extract weight deltas."""
    from qwen_hydra.extract import extract as run_extract

    click.echo(f"Extracting Qwen3-{size} deltas to {output} ...")
    manifest = run_extract(
        output_dir=output,
        size=size,
        cache_dir=cache_dir,
        sparsity_threshold=threshold,
    )

    click.echo()
    click.echo("Extraction complete!")
    click.echo(f"  Trunk: {manifest['trunk_size_bytes'] / 1e6:.1f} MB")
    for task_name, info in manifest["tasks"].items():
        delta_mb = info["delta_size_bytes"] / 1e6
        click.echo(f"  {task_name:10s}: {delta_mb:.2f} MB ({info['num_delta_params']} params)")

    total = manifest["trunk_size_bytes"] + sum(
        t["delta_size_bytes"] for t in manifest["tasks"].values()
    )
    click.echo(f"  Total: {total / 1e6:.1f} MB")


@main.command()
@click.option(
    "--dir", "-d",
    "extracted_dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to extracted deltas directory.",
)
def info(extracted_dir: Path):
    """Show delta sizes, memory stats, and manifest contents."""
    manifest_path = extracted_dir / "manifest.json"
    if not manifest_path.exists():
        click.echo(f"No manifest.json found in {extracted_dir}", err=True)
        raise SystemExit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    click.echo(f"Qwen-Hydra Delta Directory: {extracted_dir}")
    click.echo(f"  Size:       {manifest.get('size', 'unknown')}")
    click.echo(f"  Base model: {manifest['base_model']}")
    click.echo(f"  Trunk:      {manifest['trunk_size_bytes'] / 1e6:.1f} MB")
    click.echo()

    for task_name, info in manifest["tasks"].items():
        delta_mb = info["delta_size_bytes"] / 1e6
        delta_file = info.get("delta_file", "—")
        click.echo(f"  [{task_name:8s}]  {delta_mb:7.2f} MB  ({info['num_delta_params']} params)  {delta_file}")

    total = manifest["trunk_size_bytes"] + sum(
        t["delta_size_bytes"] for t in manifest["tasks"].values()
    )
    click.echo()
    click.echo(f"  Total on disk: {total / 1e6:.1f} MB")
    click.echo(f"  vs 3x separate: ~{manifest['trunk_size_bytes'] * 3 / 1e6:.0f} MB")
    savings = (1 - total / (manifest["trunk_size_bytes"] * 3)) * 100
    click.echo(f"  Savings: ~{savings:.0f}%")


if __name__ == "__main__":
    main()
