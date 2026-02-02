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
    --downsample-factor 1.0 \
    --augment-train
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
    # Enable data augmentation for training output only.
    parser.add_argument(
        "--augment-train",
        action="store_true",
        help="Augment training images with rotations and flip.",
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


def apply_rotation(image, k: int):
    """Rotate an image by 90 degrees k times.

    Args:
        image: Image array to rotate.
        k (int): Number of 90-degree rotations (1, 2, or 3).

    Returns:
        Rotated image array.
    """
    return cv2.rotate(image, {1: cv2.ROTATE_90_CLOCKWISE,
                              2: cv2.ROTATE_180,
                              3: cv2.ROTATE_90_COUNTERCLOCKWISE}[k])


def apply_flip(image):
    """Flip an image horizontally.

    Args:
        image: Image array to flip.

    Returns:
        Flipped image array.
    """
    return cv2.flip(image, 1)


def build_augmented_targets(base_path: Path, suffix: str) -> Path:
    """Create an augmented output path with a suffix.

    Args:
        base_path (Path): Original output path.
        suffix (str): Suffix for augmentation.

    Returns:
        Path: Output path with suffix inserted before extension.
    """
    return base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")


def generate_transform_plan(height: int, width: int) -> list[str]:
    """Return a list of augmentation tags without mathematical duplicates.

    Args:
        height (int): Image height.
        width (int): Image width.

    Returns:
        list[str]: List of transform tags to apply.
    """
    transforms = ["rot90", "rot180", "rot270", "flip"]
    # All listed transforms are mathematically distinct for any image size.
    return transforms


def write_augmented_pair(
    image,
    mask,
    image_target: Path,
    mask_target: Path,
    transform: str,
) -> None:
    """Write an augmented image/mask pair with consistent naming.

    Args:
        image: Image array to transform.
        mask: Mask array to transform.
        image_target (Path): Base image output path.
        mask_target (Path): Base mask output path.
        transform (str): Transform tag.

    Returns:
        None
    """
    if transform == "flip":
        aug_image = apply_flip(image)
        aug_mask = apply_flip(mask)
    else:
        k = {"rot90": 1, "rot180": 2, "rot270": 3}[transform]
        aug_image = apply_rotation(image, k)
        aug_mask = apply_rotation(mask, k)
    aug_image_path = build_augmented_targets(image_target, transform)
    aug_mask_path = build_augmented_targets(mask_target, transform)
    if aug_image_path.exists() or aug_mask_path.exists():
        raise FileExistsError(
            f"Destination already exists: {aug_image_path} or {aug_mask_path}"
        )
    if not cv2.imwrite(str(aug_image_path), aug_image):
        raise ValueError(f"Failed to write image: {aug_image_path}")
    if not cv2.imwrite(str(aug_mask_path), aug_mask):
        raise ValueError(f"Failed to write image: {aug_mask_path}")


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
    augment: bool,
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
        if downsample_factor == 1.0:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
            if mask is None:
                raise ValueError(f"Failed to read image: {mask_path}")
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
            if mask is None:
                raise ValueError(f"Failed to read image: {mask_path}")
            image = resize_array(image, downsample_factor, is_mask=False)
            mask = resize_array(mask, downsample_factor, is_mask=True)
        if not cv2.imwrite(str(image_target), image):
            raise ValueError(f"Failed to write image: {image_target}")
        if not cv2.imwrite(str(mask_target), mask):
            raise ValueError(f"Failed to write image: {mask_target}")
        # Apply training-only augmentation with consistent naming.
        if augment:
            transforms = generate_transform_plan(image.shape[0], image.shape[1])
            for transform in transforms:
                write_augmented_pair(
                    image, mask, image_target, mask_target, transform
                )


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
    copy_pairs(train_pairs, args.train_dir, args.downsample_factor, args.augment_train)
    copy_pairs(val_pairs, args.val_dir, args.downsample_factor, False)


if __name__ == "__main__":
    main()
