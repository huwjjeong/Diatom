import hashlib
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_VERSION = "file_split_v3_rot_m20"


def color_jitter_fixed(img: Image.Image) -> Image.Image:
    out = ImageEnhance.Brightness(img).enhance(1.10)
    out = ImageEnhance.Contrast(out).enhance(1.10)
    out = ImageEnhance.Color(out).enhance(1.05)
    return out


def save_train_augmentations(img_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem
    ext = img_path.suffix.lower()
    if ext not in IMG_EXTS:
        ext = ".png"

    with Image.open(img_path) as im0:
        im = im0.convert("RGB")
        variants = {
            "orig": im,
            "hflip": ImageOps.mirror(im),
            "rot20": im.rotate(20, resample=Image.BILINEAR),
            "rot_m20": im.rotate(-20, resample=Image.BILINEAR),
            "cj": color_jitter_fixed(im),
            "gblur3": im.filter(ImageFilter.GaussianBlur(radius=1.0)),
        }

        for tag, variant in variants.items():
            variant.save(out_dir / f"{stem}__{tag}{ext}")


def source_fingerprint(src_root: Path) -> str:
    h = hashlib.sha256()
    for img_path in sorted(
        p for p in src_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ):
        rel = img_path.relative_to(src_root).as_posix()
        stat = img_path.stat()
        h.update(rel.encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(stat.st_mtime_ns).encode("utf-8"))
    return h.hexdigest()


def species_from_flat_filename(genus: str, img_path: Path) -> str:
    label = img_path.stem.split("_", 1)[0].strip()
    prefix = f"{genus} "
    if label.startswith(prefix):
        species = label[len(prefix):].strip()
    else:
        parts = label.split()
        species = " ".join(parts[1:]).strip() if len(parts) > 1 else label
    return species or genus


def split_species_folder_name(folder_name: str):
    parts = folder_name.split()
    if len(parts) >= 2:
        return parts[0], " ".join(parts[1:])
    return folder_name, folder_name


def iter_species_groups(src_root: Path):
    for genus_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        species_dirs = sorted(p for p in genus_dir.iterdir() if p.is_dir())
        direct_images = sorted(
            p for p in genus_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )

        if species_dirs:
            for species_dir in species_dirs:
                images = sorted(
                    p for p in species_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMG_EXTS
                )
                if images:
                    yield genus_dir.name, species_dir.name, images

        if direct_images:
            if " " in genus_dir.name:
                genus, species = split_species_folder_name(genus_dir.name)
                yield genus, species, direct_images
                continue

            grouped = {}
            for img_path in direct_images:
                species = species_from_flat_filename(genus_dir.name, img_path)
                grouped.setdefault(species, []).append(img_path)
            for species, images in sorted(grouped.items()):
                yield genus_dir.name, species, sorted(images)


def ensure_fixed_split_aug(
    source_orig_dir: str,
    split_dir: str,
    test_ratio: float,
    seed: int,
    overwrite: bool = False,
):
    src_root = Path(source_orig_dir)
    if not src_root.is_dir():
        raise FileNotFoundError(f"SOURCE_ORIG_DIR not found: {src_root}")

    split_root = Path(split_dir)
    train_orig_root = split_root / "train_orig"
    test_orig_root = split_root / "test_orig"
    train_aug_root = split_root / "train_aug"
    marker_path = split_root / ".split_version"
    fingerprint_path = split_root / ".source_fingerprint"
    current_fingerprint = source_fingerprint(src_root)

    split_exists = (
        train_orig_root.exists()
        and test_orig_root.exists()
        and train_aug_root.exists()
        and marker_path.exists()
        and marker_path.read_text().strip() == SPLIT_VERSION
        and fingerprint_path.exists()
        and fingerprint_path.read_text().strip() == current_fingerprint
    )
    if split_exists and not overwrite:
        print(f"[split] using existing split: {split_root}")
        return

    if split_root.exists():
        print(f"[split] rebuilding split because source data or split version changed: {split_root}")
        shutil.rmtree(split_root)

    rng = random.Random(seed)
    total_train = 0
    total_test = 0
    total_aug = 0

    for genus, species, images in iter_species_groups(src_root):
        if len(images) == 0:
            continue

        rng.shuffle(images)
        if len(images) <= 1:
            n_test = 0
        else:
            n_test = max(1, int(round(len(images) * test_ratio)))
            n_test = min(n_test, len(images) - 1)

        test_images = images[:n_test]
        train_images = images[n_test:]
        rel_dir = Path(genus) / species

        for img_path in train_images:
            dst = train_orig_root / rel_dir / img_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dst)
            save_train_augmentations(img_path, train_aug_root / rel_dir)
            total_train += 1
            total_aug += 6

        for img_path in test_images:
            dst = test_orig_root / rel_dir / img_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dst)
            total_test += 1

    marker_path.write_text(SPLIT_VERSION + "\n")
    fingerprint_path.write_text(current_fingerprint + "\n")
    print(f"[split] built: {split_root}")
    print(f"[split] train_orig={total_train} | train_aug={total_aug} | test_orig={total_test}")
