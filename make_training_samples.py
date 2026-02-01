#!/usr/bin/env python3
"""
Split training images into train/validation directories, preserving samples.

Usage:
  python make_training_samples.py \
    --source /workspaces/mito-counter/training_data \
    --train-dir /workspaces/mito-counter/training_samples/train \
    --val-dir /workspaces/mito-counter/training_samples/validation \
    --ratio 0.2 \
    --seed 42 \
    --downsample-factor 1.0
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the split and downsampling script.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Copy image/mask pairs from condition subdirectories into train/"
            "validation directories, preserving sample folders."
        )
    )
    # Source directory containing condition folders with sample subfolders.
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/workspaces/mito-counter/training_data"),
        help="Source directory containing condition folders.",
    )
    # Output directory for training samples.
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("/workspaces/mito-counter/training_samples/train"),
        help="Destination directory for training images.",
    )
    # Output directory for validation samples.
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=Path("/workspaces/mito-counter/training_samples/validation"),
        help="Destination directory for validation images.",
    )
    # Fraction of pairs to put in validation.
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (0 < ratio < 1).",
    )
    # Seed to make splits repeatable.
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splits.",
    )
    # Downsample factor applied to images and masks.
    parser.add_argument(
        "--downsample-factor",
        type=float,
        default=1.0,
        help="Downsample images/masks by factor (>= 1.0).",
    )
    return parser.parse_args()


def list_samples(source_dir: Path) -> list[Path]:
    """Return sample directories under each condition folder in source.

    Args:
        source_dir (Path): Root directory containing condition folders.

    Returns:
        list[Path]: Sample directory paths.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {source_dir}")

    # Collect each sample folder under any condition directory.
    condition_dirs = [p for p in source_dir.iterdir() if p.is_dir()]
    samples: list[Path] = []
    for condition in condition_dirs:
        for sample in condition.iterdir():
            if sample.is_dir():
                samples.append(sample)
    return samples


def validate_samples(samples: list[Path]) -> None:
    """Ensure sample folder names are unique after flattening.

    Args:
        samples (list[Path]): Sample directories to validate.

    Returns:
        None
    """
    # Duplicate sample names would collide in output folders.
    names = [sample.name for sample in samples]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        dup_list = ", ".join(sorted(duplicates))
        raise ValueError(
            "Duplicate sample folder names found after flattening: "
            f"{dup_list}. Rename samples or keep condition folders."
        )


def list_image_mask_pairs(samples: list[Path]) -> list[tuple[Path, Path, Path]]:
    """Return (sample_dir, image_path, mask_path) tuples for all pairs.

    Args:
        samples (list[Path]): Sample directories to scan for pairs.

    Returns:
        list[tuple[Path, Path, Path]]: Sample, image, and mask paths.
    """
    pairs: list[tuple[Path, Path, Path]] = []
    for sample in samples:
        # Each sample must contain images/ and masks/ directories.
        images_dir = sample / "images"
        masks_dir = sample / "masks"
        if not images_dir.is_dir() or not masks_dir.is_dir():
            raise FileNotFoundError(
                f"Expected 'images' and 'masks' folders in {sample}"
            )
        # Pair each image with a mask of the same relative path/name.
        image_files = [p for p in images_dir.rglob("*") if p.is_file()]
        for image_path in image_files:
            rel_path = image_path.relative_to(images_dir)
            mask_path = masks_dir / rel_path
            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing mask for image: {image_path} (expected {mask_path})"
                )
            pairs.append((sample, image_path, mask_path))
    return pairs


def split_samples(
    samples: list[tuple[Path, Path, Path]],
    ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, Path, Path]], list[tuple[Path, Path, Path]]]:
    """Shuffle and split pairs into train and validation lists.

    Args:
        samples (list[tuple[Path, Path, Path]]): Sample/image/mask triples.
        ratio (float): Validation split ratio between 0 and 1.
        seed (int): Random seed for deterministic shuffling.

    Returns:
        tuple[list[tuple[Path, Path, Path]], list[tuple[Path, Path, Path]]]:
            Train pairs and validation pairs.
    """
    if not 0 < ratio < 1:
        raise ValueError(f"Ratio must be between 0 and 1 (exclusive). Got {ratio}.")
    # Deterministic shuffle for repeatable splits.
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    # Compute how many pairs go to validation.
    val_count = int(len(shuffled) * ratio)
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]
    return train_samples, val_samples


def resize_array(image, factor: float, is_mask: bool):
    """Resize an image or mask by the given factor.

    Args:
        image: Image array read from disk.
        factor (float): Downsample factor (>= 1.0).
        is_mask (bool): Whether the image is a mask.

    Returns:
        Resized image array.
    """
    if factor == 1.0:
        return image
    height, width = image.shape[:2]
    # Compute new size with a minimum of 1 pixel.
    new_width = max(1, int(round(width / factor)))
    new_height = max(1, int(round(height / factor)))
    # Use nearest-neighbor for masks to preserve labels.
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(image, (new_width, new_height), interpolation=interp)


def copy_or_downsample(
    source: Path,
    destination: Path,
    factor: float,
    is_mask: bool,
) -> None:
    """Copy a file or downsample and write it to destination.

    Args:
        source (Path): Source image path.
        destination (Path): Destination image path.
        factor (float): Downsample factor (>= 1.0).
        is_mask (bool): Whether the image is a mask.

    Returns:
        None
    """
    if factor == 1.0:
        # No resizing requested; just copy.
        shutil.copy2(source, destination)
        return
    # Load the image, resize, and save to destination.
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {source}")
    resized = resize_array(image, factor, is_mask=is_mask)
    if not cv2.imwrite(str(destination), resized):
        raise ValueError(f"Failed to write image: {destination}")


def copy_pairs(
    pairs: list[tuple[Path, Path, Path]],
    destination: Path,
    downsample_factor: float,
) -> None:
    """Copy (and optionally downsample) each image/mask pair into destination.

    Args:
        pairs (list[tuple[Path, Path, Path]]): Sample/image/mask triples.
        destination (Path): Output root directory.
        downsample_factor (float): Downsample factor (>= 1.0).

    Returns:
        None
    """
    # Create the root output directory.
    destination.mkdir(parents=True, exist_ok=True)
    for sample_dir, image_path, mask_path in pairs:
        # Preserve sample folder name and relative paths under it.
        image_rel = image_path.relative_to(sample_dir)
        mask_rel = mask_path.relative_to(sample_dir)
        image_target = destination / sample_dir.name / image_rel
        mask_target = destination / sample_dir.name / mask_rel
        if image_target.exists() or mask_target.exists():
            raise FileExistsError(
                f"Destination already exists: {image_target} or {mask_target}"
            )
        # Ensure parent directories exist before writing.
        image_target.parent.mkdir(parents=True, exist_ok=True)
        mask_target.parent.mkdir(parents=True, exist_ok=True)
        # Copy or downsample image and mask consistently.
        copy_or_downsample(
            image_path, image_target, downsample_factor, is_mask=False
        )
        copy_or_downsample(mask_path, mask_target, downsample_factor, is_mask=True)


def main() -> None:
    """Run the split workflow from CLI arguments.

    Returns:
        None
    """
    # Parse CLI parameters.
    args = parse_args()
    # Discover samples under condition directories.
    samples = list_samples(args.source)
    if not samples:
        raise ValueError(f"No sample directories found under {args.source}.")
    # Ensure sample names are unique after flattening.
    validate_samples(samples)
    # Collect image/mask pairs.
    pairs = list_image_mask_pairs(samples)
    if not pairs:
        raise ValueError(f"No image/mask pairs found under {args.source}.")
    # Split pairs into train and validation.
    train_pairs, val_pairs = split_samples(pairs, args.ratio, args.seed)
    if args.downsample_factor < 1.0:
        raise ValueError("Downsample factor must be >= 1.0.")
    # Copy and optionally downsample into the output directories.
    copy_pairs(train_pairs, args.train_dir, args.downsample_factor)
    copy_pairs(val_pairs, args.val_dir, args.downsample_factor)


if __name__ == "__main__":
    main()
