# LR = 3e-4
# ALPHA = 0.5
# TEMPERATURE = 4.0
# LORA_R = 8
# LORA_ALPHA = 16
# LORA_DROPOUT = 0.1
# TEACHER_EPOCHS = 30
# TEACHER_LABEL_SMOOTH = 0.2
# TEACHER_WEIGHT_DECAY = 0.05



import os, random, time
import numpy as np
from collections import defaultdict

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from torchmetrics.classification import MulticlassAccuracy
from peft import LoraConfig, get_peft_model

from PIL import Image, ImageFile
from fixed_split_aug import ensure_fixed_split_aug

# Keep truncated JPEGs readable instead of failing DataLoader workers.
ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================
# 0) Paths 
# =========================================================
SOURCE_ORIG_DIR = "/home/woojin/바탕화면/model/model/7_ath"
SPLIT_DIR = "/home/woojin/바탕화면/model/7_ath_split_aug"
ORIG_DIR = os.path.join(SPLIT_DIR, "train_orig")
TEST_DIR = os.path.join(SPLIT_DIR, "test_orig")
AUG_DIR  = os.path.join(SPLIT_DIR, "train_aug")

EXP_DIR      = "./exp_aug_kd_swin_fixedsplit"
TEACHER_DIR  = os.path.join(EXP_DIR, "teachers")
SOFT_DIR     = os.path.join(EXP_DIR, "softcache")
CKPT_DIR     = os.path.join(EXP_DIR, "checkpoints")

os.makedirs(TEACHER_DIR, exist_ok=True)
os.makedirs(SOFT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

SOFT_CACHE_PATH = os.path.join(SOFT_DIR, "softlabel_cache_train_mix_keep_teacher_resnet50_resize224.pt")


# =========================================================
# 1) Config
# =========================================================
BATCH_SIZE = 16
NUM_WORKERS = 8

EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 0.01
SEED = 42

# KD hyperparams
ALPHA = 0.5
TEMPERATURE = 4.0

KD_ONLY_AUGMENTED = False

# Student / Teacher backbone
STUDENT_MODEL_NAME = "swin_small_patch4_window7_224"
TEACHER_MODEL_NAME = "resnet50"

# LoRA
USE_LORA_STUDENT = True
USE_LORA_TEACHER = True
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# student / teacher target modules
LORA_TARGET_MODULES_SWIN = ["qkv"]
LORA_TARGET_MODULES_RESNET = ["fc"]  

# split (FINAL TEST split)
MIN_TEST_PER_GENUS = 5
TEST_RATIO = 0.2

# Teacher split / train
TEACHER_VAL_RATIO = 0.2
MIN_TEACHER_VAL_PER_GENUS = 20
TEACHER_EPOCHS = 30
TEACHER_LABEL_SMOOTH = 0.2
TEACHER_WEIGHT_DECAY = 0.05

# cache dtype
SOFTCACHE_DTYPE = torch.float16

# fixed image size
IMG_SIZE = 224

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


# =========================================================
# 2) Reproducibility
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)


# =========================================================
# 3) Transforms (fixed-size: Resize to 224x224)
# =========================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# =========================================================
# 4) Dataset split (safe split: test는 원본 only)
# =========================================================
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

ensure_fixed_split_aug(
    source_orig_dir=SOURCE_ORIG_DIR,
    split_dir=SPLIT_DIR,
    test_ratio=TEST_RATIO,
    seed=SEED,
)

class SpeciesFolderDataset(Dataset):
    """
    Reads a two-level layout:
      root/genus/species/image.ext
    and uses species as class labels (global class name: "genus/species").
    """
    def __init__(
        self,
        root: str,
        transform=None,
        class_to_idx: dict = None,
        allow_missing_classes: bool = False,
    ):
        self.root = root
        self.transform = transform
        self.loader = default_loader
        self.samples = []
        self.allow_missing_classes = allow_missing_classes

        if class_to_idx is None:
            classes = self._discover_classes(root)
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = dict(class_to_idx)

        self.classes = [None] * len(self.class_to_idx)
        for cname, cidx in self.class_to_idx.items():
            self.classes[cidx] = cname

        self._gather_samples()
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under: {root}")
        self.targets = [y for _, y in self.samples]

    @staticmethod
    def _discover_classes(root: str):
        classes = []
        for genus in sorted(os.listdir(root)):
            genus_dir = os.path.join(root, genus)
            if not os.path.isdir(genus_dir):
                continue
            for species in sorted(os.listdir(genus_dir)):
                species_dir = os.path.join(genus_dir, species)
                if not os.path.isdir(species_dir):
                    continue
                classes.append(f"{genus}/{species}")
        return classes

    def _gather_samples(self):
        seen_classes = set()
        for genus in sorted(os.listdir(self.root)):
            genus_dir = os.path.join(self.root, genus)
            if not os.path.isdir(genus_dir):
                continue
            for species in sorted(os.listdir(genus_dir)):
                species_dir = os.path.join(genus_dir, species)
                if not os.path.isdir(species_dir):
                    continue
                cname = f"{genus}/{species}"
                if cname not in self.class_to_idx:
                    raise ValueError(f"Unexpected class in {self.root}: {cname}")
                cidx = self.class_to_idx[cname]
                seen_classes.add(cname)

                for fname in sorted(os.listdir(species_dir)):
                    path = os.path.join(species_dir, fname)
                    if not os.path.isfile(path):
                        continue
                    if not fname.lower().endswith(IMG_EXTS):
                        continue
                    self.samples.append((path, cidx))

        missing = sorted(set(self.class_to_idx.keys()) - seen_classes)
        if len(missing) > 0:
            if self.allow_missing_classes:
                print(
                    f"[WARN] Missing classes in {self.root}: {len(missing)} "
                    f"(showing up to 10) {missing[:10]}"
                )
            else:
                raise ValueError(f"Missing classes in {self.root}: {missing[:10]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        x = self.loader(path)
        if self.transform is not None:
            x = self.transform(x)
        return x, y

if not os.path.exists(ORIG_DIR):
    raise FileNotFoundError(f"ORIG_DIR not found: {ORIG_DIR}")
if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"TEST_DIR not found: {TEST_DIR}")
if not os.path.exists(AUG_DIR):
    raise FileNotFoundError(f"AUG_DIR not found: {AUG_DIR}")

orig_ds = SpeciesFolderDataset(ORIG_DIR, transform=test_tf)
test_orig_ds = SpeciesFolderDataset(
    TEST_DIR,
    transform=test_tf,
    class_to_idx=orig_ds.class_to_idx,
    allow_missing_classes=True,
)
aug_ds  = SpeciesFolderDataset(
    AUG_DIR,
    transform=train_tf,
    class_to_idx=orig_ds.class_to_idx,
    allow_missing_classes=True,
)

assert orig_ds.classes == aug_ds.classes, "Class folders mismatch between ORIG_DIR and AUG_DIR"
assert orig_ds.classes == test_orig_ds.classes, "Class folders mismatch between ORIG_DIR and TEST_DIR"
classes_global = orig_ds.classes
num_classes_global = len(classes_global)

print("num_classes_global:", num_classes_global)
print("train orig images:", len(orig_ds))
print("aug  images:", len(aug_ds))
print("test orig images:", len(test_orig_ds))

# (A) 원본 샘플 key: (class_name, base)
orig_samples = orig_ds.samples
orig_keys = []
for p, y in orig_samples:
    cls = orig_ds.classes[y]
    base = os.path.splitext(os.path.basename(p))[0]
    orig_keys.append((cls, base))

# (B) fixed split is already built on disk:
#     ORIG_DIR=train_orig, AUG_DIR=train_aug, TEST_DIR=test_orig.
#     Do not split again here.
genus_to_orig_idxs = defaultdict(list)
for idx, (_, y) in enumerate(orig_ds.samples):
    genus = classes_global[y].split("/")[0]
    genus_to_orig_idxs[genus].append(idx)

train_orig_idxs = list(range(len(orig_ds)))
rng = np.random.default_rng(SEED)

train_key_set = set(orig_keys[i] for i in train_orig_idxs)
print(f"[Fixed split] train_orig={len(train_orig_idxs)} | test_orig={len(test_orig_ds)}")

# (C) aug에서 train에 해당하는 파일만 선택 (leakage 방지)
aug_keep_indices = []
for j, (p, y) in enumerate(aug_ds.samples):
    cls = aug_ds.classes[y]
    fname = os.path.splitext(os.path.basename(p))[0]  # e.g., "ABC__flip"
    base = fname.split("__")[0]
    if (cls, base) in train_key_set:
        aug_keep_indices.append(j)

print(f"aug train images selected: {len(aug_keep_indices)} (expected ~ {len(train_orig_idxs)*5})")

# Student TEST (fixed test_orig only)
test_ds = test_orig_ds


# =========================================================
# 4.25) Subset wrapper: (img, y, path)
# =========================================================
class SubsetWithPath(Dataset):
    def __init__(self, base_subset: Subset):
        if not isinstance(base_subset, Subset):
            raise TypeError("SubsetWithPath expects torch.utils.data.Subset")
        self.base_ds = base_subset.dataset
        self.indices = base_subset.indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        base_idx = self.indices[i]
        x, y = self.base_ds[base_idx]
        path, _ = self.base_ds.samples[base_idx]
        return x, y, path


# Student train = aug(train) + orig(train)
aug_train_with_path  = SubsetWithPath(Subset(aug_ds,  aug_keep_indices))
orig_train_with_path = SubsetWithPath(Subset(orig_ds, train_orig_idxs))
train_ds = ConcatDataset([aug_train_with_path, orig_train_with_path])

print(f"[Student train] aug={len(aug_train_with_path)} + orig={len(orig_train_with_path)} => total={len(train_ds)}")

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0),
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0),
)


# =========================================================
# 4.5) Teacher validation split (train_orig에서 genus별로 val 분리)
# =========================================================
teacher_train_orig_idxs = []
teacher_val_orig_idxs = []

train_orig_set = set(train_orig_idxs)

for genus, idxs_g in genus_to_orig_idxs.items():
    idxs_train_g = np.array([i for i in idxs_g if i in train_orig_set], dtype=int)
    if len(idxs_train_g) == 0:
        continue

    rng.shuffle(idxs_train_g)
    n_total = len(idxs_train_g)

    n_val = max(int(n_total * TEACHER_VAL_RATIO), MIN_TEACHER_VAL_PER_GENUS)

    if n_total <= 1:
        n_val = 0
    else:
        n_val = min(n_val, n_total - 1)

    teacher_val_orig_idxs.extend(idxs_train_g[:n_val].tolist())
    teacher_train_orig_idxs.extend(idxs_train_g[n_val:].tolist())

print(f"[Teacher split] teacher_train_orig={len(teacher_train_orig_idxs)} | teacher_val_orig={len(teacher_val_orig_idxs)}")
print(f"[Teacher split] MIN_TEACHER_VAL_PER_GENUS={MIN_TEACHER_VAL_PER_GENUS} (if possible)")


# =========================================================
# 5) Genus mapping (global class idx <-> genus local idx)
# =========================================================
def get_genus(name: str) -> str:
    return name.split("/")[0]

genus_to_global_idxs = defaultdict(list)
for gi, cname in enumerate(classes_global):
    genus_to_global_idxs[get_genus(cname)].append(gi)

all_genera = sorted(genus_to_global_idxs.keys())
print("num genera found:", len(all_genera))
print("example genera:", all_genera[:10])

# Use all genera discovered from dataset instead of a hardcoded list.
TARGET_GENERA = all_genera[:]

genus_local_to_global = {}
for g in TARGET_GENERA:
    glist = genus_to_global_idxs[g]
    genus_local_to_global[g] = glist[:]

global_to_genus = {gi: get_genus(cname) for gi, cname in enumerate(classes_global)}


# =========================================================
# 6) Model builders
# =========================================================
def apply_lora_and_set_trainable(model, model_name: str):
    if "swin" in model_name:
        target_modules = LORA_TARGET_MODULES_SWIN
    elif "resnet" in model_name:
        target_modules = LORA_TARGET_MODULES_RESNET
    else:
        raise ValueError(f"Unsupported model for LoRA: {model_name}")

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True

    base_model = model.base_model.model

    if hasattr(base_model, "head") and base_model.head is not None:
        for p in base_model.head.parameters():
            p.requires_grad = True

    if hasattr(base_model, "fc") and base_model.fc is not None:
        for p in base_model.fc.parameters():
            p.requires_grad = True

    return model

def build_backbone(model_name: str, num_classes: int, use_lora: bool):
    if "swin" in model_name:
        m = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
        )
    else:
        m = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
        )

    if use_lora:
        m = apply_lora_and_set_trainable(m, model_name)

    return m


# =========================================================
# 7) Teacher train/save/load (genus experts)
#    - train: aug(train) + orig(train: teacher_train_orig_idxs)
#    - val:   orig(val: teacher_val_orig_idxs)
# =========================================================
def teacher_ckpt_path(genus: str, n_local: int):
    return os.path.join(
        TEACHER_DIR,
        f"teacher_{TEACHER_MODEL_NAME}_{genus}_ncls{n_local}.pt"
    )

def train_teacher_for_genus(genus: str, epochs: int = TEACHER_EPOCHS):
    glist = genus_to_global_idxs[genus]
    n_local = len(glist)
    if n_local <= 1:
        print(
            f"\n[Teacher Skip] {genus} | local classes={n_local} -> "
            "skip training (trivial single-class, assumed 100.00%)"
        )
        return
    global_to_local = {glob: loc for loc, glob in enumerate(glist)}

    keep_train_aug = []
    for j in aug_keep_indices:
        _, y = aug_ds.samples[j]
        if get_genus(classes_global[y]) == genus:
            keep_train_aug.append(j)
    train_subset_aug = Subset(aug_ds, keep_train_aug)

    keep_train_orig = []
    for idx in teacher_train_orig_idxs:
        _, y = orig_ds.samples[idx]
        if get_genus(classes_global[y]) == genus:
            keep_train_orig.append(idx)
    train_subset_orig = Subset(orig_ds, keep_train_orig)

    train_subset = ConcatDataset([train_subset_aug, train_subset_orig])

    keep_val = []
    for idx in teacher_val_orig_idxs:
        _, y = orig_ds.samples[idx]
        if get_genus(classes_global[y]) == genus:
            keep_val.append(idx)
    val_subset = Subset(orig_ds, keep_val)

    train_loader_g = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader_g = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    teacher = build_backbone(
        TEACHER_MODEL_NAME,
        num_classes=n_local,
        use_lora=USE_LORA_TEACHER
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=TEACHER_LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, teacher.parameters()),
        lr=LR,
        weight_decay=TEACHER_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    @torch.no_grad()
    def eval_teacher():
        teacher.eval()
        total, correct = 0, 0
        for x, y_global in val_loader_g:
            x = x.to(device, non_blocking=True)
            y_local = torch.tensor(
                [global_to_local[int(g)] for g in y_global.tolist()],
                device=device,
                dtype=torch.long
            )
            pred = teacher(x).argmax(dim=1)
            correct += (pred == y_local).sum().item()
            total += y_local.numel()
        return correct / max(1, total)

    best = 0.0
    save_path = teacher_ckpt_path(genus, n_local)

    print(
        f"\n[Teacher Train] {genus} | model={TEACHER_MODEL_NAME} | local classes={n_local} | "
        f"train={len(train_subset)} (aug {len(train_subset_aug)} + orig {len(train_subset_orig)}) | "
        f"val(orig)={len(val_subset)}"
    )

    for ep in range(1, epochs + 1):
        teacher.train()
        total, correct = 0, 0

        for x, y_global in train_loader_g:
            x = x.to(device, non_blocking=True)
            y_local = torch.tensor(
                [global_to_local[int(g)] for g in y_global.tolist()],
                device=device,
                dtype=torch.long
            )

            optimizer.zero_grad()
            logits = teacher(x)
            loss = criterion(logits, y_local)
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            correct += (pred == y_local).sum().item()
            total += y_local.numel()

        scheduler.step()
        val_acc = eval_teacher()
        if val_acc > best:
            best = val_acc
            torch.save(teacher.state_dict(), save_path)

        print(
            f"[Teacher {genus}] ep {ep:03d}/{epochs} | "
            f"train {correct/max(1,total)*100:.2f}% | "
            f"val {val_acc*100:.2f}% | best {best*100:.2f}%"
        )

    print("saved:", save_path)

def load_or_train_teachers():
    teachers = {}
    for g in TARGET_GENERA:
        n_local = len(genus_to_global_idxs[g])
        if n_local <= 1:
            teachers[g] = None
            print(
                f"[Teacher Skip] {g} | local classes={n_local} -> "
                "skip load/train (assumed 100.00%)"
            )
            continue

        ckpt = teacher_ckpt_path(g, n_local)
        if not os.path.exists(ckpt):
            print(f"[No ckpt] train teacher for {g}")
            train_teacher_for_genus(g, epochs=TEACHER_EPOCHS)

        tmodel = build_backbone(
            TEACHER_MODEL_NAME,
            num_classes=n_local,
            use_lora=USE_LORA_TEACHER
        ).to(device)

        try:
            tmodel.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
        except Exception:
            print(f"[Teacher Rebuild] {g} | checkpoint mismatch -> retrain")
            train_teacher_for_genus(g, epochs=TEACHER_EPOCHS)
            tmodel.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
        tmodel.eval()
        for p in tmodel.parameters():
            p.requires_grad = False

        teachers[g] = tmodel
        print(f"Loaded teacher[{g}] from {ckpt}")

    return teachers

teachers = load_or_train_teachers()


# =========================================================
# 8) Student model (global)
# =========================================================
student = build_backbone(
    STUDENT_MODEL_NAME,
    num_classes=num_classes_global,
    use_lora=USE_LORA_STUDENT
).to(device)

print("\n[Student trainables]")
try:
    student.print_trainable_parameters()
except Exception:
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    print(f"trainable={trainable} / total={total} ({trainable/total*100:.2f}%)")
    for name, p in student.named_parameters():
        if p.requires_grad:
            print("  ", name)


# =========================================================
# 9) KD utilities
# =========================================================
def pad_teacher_logits_to_global(g: str, t_logits_local: torch.Tensor, device):
    m, _ = t_logits_local.shape
    t_global = torch.zeros((m, num_classes_global), device=device, dtype=t_logits_local.dtype)
    global_idxs = genus_local_to_global[g]
    idx_tensor = torch.tensor(global_idxs, device=device, dtype=torch.long)
    t_global[:, idx_tensor] = t_logits_local
    return t_global

def kd_kl_div(student_logits, teacher_logits, T=4.0):
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)

def is_orig_from_filename(path: str) -> bool:
    base = os.path.splitext(os.path.basename(path))[0]
    return "__orig" in base


# =========================================================
# 10) Build / Load image-wise soft-label cache (train subset only)
# =========================================================
@torch.no_grad()
def build_softlabel_cache_for_train_subset(
    teachers: dict,
    train_subset_with_path: Dataset,
    save_path: str,
    batch_size: int = 64,
    num_workers: int = 2,
    device: str = "cuda",
    store_dtype: torch.dtype = torch.float16,
):
    tmp_loader = DataLoader(
        train_subset_with_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    path2logits = {}
    seen = 0

    for x, y_global, paths in tmp_loader:
        x = x.to(device, non_blocking=True)
        y_global = y_global.to(device, non_blocking=True)

        idxs_by_genus = defaultdict(list)
        for bi, yi in enumerate(y_global.tolist()):
            g = global_to_genus[int(yi)]
            if (g in teachers) and (teachers[g] is not None):
                if KD_ONLY_AUGMENTED and is_orig_from_filename(paths[bi]):
                    continue
                idxs_by_genus[g].append(bi)

        for g, bidxs in idxs_by_genus.items():
            bx = x[bidxs]
            t_local = teachers[g](bx)
            t_global = pad_teacher_logits_to_global(g, t_local, device=device)
            t_global = t_global.detach().to("cpu", dtype=store_dtype)

            for k, bi in enumerate(bidxs):
                p = paths[bi]
                path2logits[p] = t_global[k].contiguous()

        seen += len(paths)
        if seen % 2000 == 0:
            print(f"[softcache] processed {seen}/{len(train_subset_with_path)}")

    torch.save({
        "path2logits": path2logits,
        "num_classes_global": num_classes_global,
        "dtype": str(store_dtype),
        "kd_only_augmented": KD_ONLY_AUGMENTED,
        "teacher_model_name": TEACHER_MODEL_NAME,
        "note": "teacher logits padded to global space. keys are file paths in student train subset (aug+orig)."
    }, save_path)

    print("\n[softcache] DONE")
    print("saved:", save_path)
    print("cached items:", len(path2logits), "/", len(train_subset_with_path))
    return path2logits

def load_or_build_softcache():
    if os.path.exists(SOFT_CACHE_PATH):
        obj = torch.load(SOFT_CACHE_PATH, map_location="cpu")
        mismatch = False
        if obj.get("num_classes_global", None) != num_classes_global:
            print("[softcache] num_classes_global changed -> rebuilding cache")
            mismatch = True
        if obj.get("kd_only_augmented", None) != KD_ONLY_AUGMENTED:
            print("[softcache] KD_ONLY_AUGMENTED changed -> rebuilding cache")
            mismatch = True
        if obj.get("teacher_model_name", None) != TEACHER_MODEL_NAME:
            print("[softcache] teacher model changed -> rebuilding cache")
            mismatch = True

        if not mismatch:
            print("[softcache] loaded:", SOFT_CACHE_PATH, "items:", len(obj["path2logits"]))
            return obj["path2logits"]

    print("[softcache] not found(or mismatch), building:", SOFT_CACHE_PATH)
    return build_softlabel_cache_for_train_subset(
        teachers=teachers,
        train_subset_with_path=train_ds,
        save_path=SOFT_CACHE_PATH,
        batch_size=max(64, BATCH_SIZE),
        num_workers=NUM_WORKERS,
        device=device,
        store_dtype=SOFTCACHE_DTYPE,
    )

softcache_path2logits = load_or_build_softcache()


# =========================================================
# 11) Train loop (CE + KD from softcache)
# =========================================================
criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, student.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

acc_metric = MulticlassAccuracy(num_classes=num_classes_global).to(device)

def train_one_epoch_kd_softcache(student, loader, softcache, alpha=0.5, T=4.0):
    student.train()
    acc_metric.reset()

    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    steps = 0

    for x, y, paths in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        s_logits = student(x)
        loss_ce = criterion_ce(s_logits, y)

        kd_indices = []
        t_list = []

        for i, p in enumerate(paths):
            if KD_ONLY_AUGMENTED and is_orig_from_filename(p):
                continue
            t = softcache.get(p, None)
            if t is None:
                continue
            kd_indices.append(i)
            t_list.append(t)

        if len(kd_indices) > 0:
            t_kd = torch.stack(t_list, dim=0).to(device, dtype=s_logits.dtype)
            s_kd = s_logits[kd_indices]
            loss_kd = kd_kl_div(s_kd, t_kd, T=T)
        else:
            loss_kd = torch.tensor(0.0, device=device)

        loss = (1 - alpha) * loss_ce + alpha * loss_kd
        loss.backward()
        optimizer.step()

        acc_metric.update(s_logits, y)

        total_loss += loss.item()
        total_ce += loss_ce.item()
        total_kd += loss_kd.item()
        steps += 1

    return {
        "loss": total_loss / max(1, steps),
        "ce": total_ce / max(1, steps),
        "kd": total_kd / max(1, steps),
        "acc": acc_metric.compute().item(),
    }

@torch.no_grad()
def evaluate(student, loader):
    student.eval()
    acc_metric.reset()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = student(x)
        acc_metric.update(logits, y)
    return acc_metric.compute().item()

SAVE_STUDENT = os.path.join(CKPT_DIR, "student_swin_small_kd_multiteacher_softcache_resize224_teacher_resnet50.pt")

best_test = 0.0
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr = train_one_epoch_kd_softcache(
        student,
        train_loader,
        softcache_path2logits,
        alpha=ALPHA,
        T=TEMPERATURE
    )
    scheduler.step()
    te_acc = evaluate(student, test_loader)

    if te_acc > best_test:
        best_test = te_acc
        torch.save(student.state_dict(), SAVE_STUDENT)

    print(
        f"[{epoch:03d}/{EPOCHS}] "
        f"loss {tr['loss']:.4f} (ce {tr['ce']:.4f}, kd {tr['kd']:.4f}) | "
        f"train acc {tr['acc']*100:.2f}% | test acc {te_acc*100:.2f}% | "
        f"best {best_test*100:.2f}% | {time.time()-t0:.1f}s"
    )

print("saved best student:", SAVE_STUDENT)


# =========================================================
# 12) Final evaluation (overall + genus-wise) on FINAL TEST (orig only)
# =========================================================
@torch.no_grad()
def evaluate_test_overall_and_genus(student, loader, classes_local):
    student.eval()
    total, correct = 0, 0
    species_correct = defaultdict(int)
    species_total = defaultdict(int)
    species_to_genus = {}

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        preds = student(x).argmax(dim=1)

        for gt, pred in zip(y, preds):
            gt = gt.item()
            pred = pred.item()
            total += 1
            ok = int(gt == pred)
            correct += ok

            species_name = classes_local[gt]
            genus = get_genus(species_name)
            species_to_genus[species_name] = genus
            species_total[species_name] += 1
            species_correct[species_name] += ok

    overall = correct / total if total > 0 else 0.0
    genus_species_accs = defaultdict(list)
    for sname, n in species_total.items():
        s_acc = species_correct[sname] / max(1, n)
        genus_species_accs[species_to_genus[sname]].append(s_acc)
    genus_acc = {g: float(np.mean(accs)) for g, accs in genus_species_accs.items()}
    return overall, genus_acc

student.load_state_dict(torch.load(SAVE_STUDENT, map_location=device), strict=True)
overall_acc, genus_acc = evaluate_test_overall_and_genus(student, test_loader, classes_global)

print("\n====================")
print("FINAL TEST (Student=Swin Small, Teacher=ResNet50+LoRA, softcache, Test=Original only)")
print("====================")
print(f"Overall Test Accuracy: {overall_acc*100:.2f}%")

single_genus_acc = []
multi_genus_acc = []
for g, acc in sorted(genus_acc.items()):
    n_species = len(genus_to_global_idxs[g])
    if n_species == 1:
        single_genus_acc.append((g, acc))
    else:
        multi_genus_acc.append((g, acc))

print("\nGenus-wise Accuracy (species=1)")
print(f"count: {len(single_genus_acc)}")
for g, acc in single_genus_acc:
    print(f"{g:15s} : {acc*100:.2f}%")

print("\nGenus-wise Accuracy (species>=2)")
print(f"count: {len(multi_genus_acc)}")
for g, acc in multi_genus_acc:
    print(f"{g:15s} : {acc*100:.2f}%")
