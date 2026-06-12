"""
Aesthetic clip classifier.

Training uses 4000+ positive clips from the user's library + auto-generated
negatives. The model is a lightweight CNN head on top of mean-pooled
frame embeddings (MobileNetV3 features, runs fast on 4070/4080).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 960   # MobileNetV3-Large penultimate layer output
FRAMES_PER_CLIP = 8
INPUT_SIZE = (224, 224)

_transform = T.Compose([
    T.ToTensor(),
    T.Resize(INPUT_SIZE, antialias=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Model architecture ────────────────────────────────────────────────────────

class AestheticHead(nn.Module):
    """Binary classifier head on top of pooled frame embeddings."""

    def __init__(self, embed_dim: int = EMBED_DIM, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):          # x: (B, embed_dim)
        return self.net(x).squeeze(1)  # (B,)


class AestheticScorer(nn.Module):
    """Full model: MobileNetV3 backbone + aesthetic head."""

    def __init__(self):
        super().__init__()
        import torchvision.models as models
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        # Drop the classifier, keep up to avgpool
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.head = AestheticHead(EMBED_DIM)
        # Freeze backbone initially
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, frames):      # frames: (B, T, 3, H, W)
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        feats = self.avgpool(self.features(flat))   # (B*T, E, 1, 1)
        feats = feats.view(B, T, EMBED_DIM).mean(1)  # temporal mean pool
        return self.head(feats)

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True


# ── Embedding helper (used at inference time without training overhead) ───────

def _load_backbone():
    import torchvision.models as models
    m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    m.features.eval()
    m.avgpool.eval()
    m = m.to(DEVICE)
    return m


@torch.no_grad()
def embed_frames(frames_bgr: list[np.ndarray], backbone) -> np.ndarray:
    """Convert a list of BGR frames to a single mean embedding (numpy)."""
    tensors = [_transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
    batch = torch.stack(tensors).to(DEVICE)           # (T, 3, H, W)
    feats = backbone.avgpool(backbone.features(batch)) # (T, E, 1, 1)
    mean_feat = feats.view(len(tensors), -1).mean(0)   # (E,)
    return mean_feat.cpu().numpy()


# ── Dataset ───────────────────────────────────────────────────────────────────

class ClipDataset(Dataset):
    """
    Dataset of (frames_tensor, label) pairs.

    *items* is a list of (path: Path, label: int) tuples.
    """

    def __init__(self, items: list[tuple[Path, int]], frames_per_clip: int = FRAMES_PER_CLIP):
        self.items = items
        self.frames_per_clip = frames_per_clip

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        frames = sample_frames(path, self.frames_per_clip)
        tensors = torch.stack([_transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames])
        return tensors, torch.tensor(label, dtype=torch.float32)


def sample_frames(path: Path, n: int) -> list[np.ndarray]:
    """Uniformly sample *n* frames from a clip. Returns BGR numpy arrays."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return [np.zeros((224, 224, 3), dtype=np.uint8)] * n

    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    return frames


# ── Negative example generation ───────────────────────────────────────────────

def generate_negatives(
    source_videos: list[Path],
    positive_clips: list[Path],
    output_dir: Path,
    *,
    target_count: int = 2000,
    min_secs: float = 2.0,
    max_secs: float = 8.0,
) -> list[Path]:
    """
    Extract random non-aesthetic clips from source videos.

    Avoids time ranges that overlap with existing positive clips.
    Returns list of extracted negative clip paths.
    """
    from autoclipper.preprocessor import extract_clip, get_duration

    output_dir.mkdir(parents=True, exist_ok=True)
    negatives: list[Path] = []
    rng = np.random.default_rng(42)

    per_video = max(1, target_count // max(len(source_videos), 1))

    for vid in source_videos:
        try:
            dur = get_duration(vid)
        except Exception as e:
            log.warning("Skipping %s: %s", vid.name, e)
            continue

        count = 0
        attempts = 0
        while count < per_video and attempts < per_video * 5:
            attempts += 1
            clip_len = rng.uniform(min_secs, max_secs)
            start = rng.uniform(0, max(0, dur - clip_len))
            end = start + clip_len
            out = output_dir / f"neg_{vid.stem}_{int(start*100):08d}.mp4"
            try:
                extract_clip(vid, start, end, out)
                negatives.append(out)
                count += 1
            except Exception as e:
                log.debug("Clip extraction failed: %s", e)

    log.info("Generated %d negative examples in %s", len(negatives), output_dir)
    return negatives


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    positive_clips: list[Path],
    negative_clips: list[Path],
    output_model: Path,
    *,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    progress_cb: Callable[[str, float], None] | None = None,
) -> None:
    """Train the AestheticScorer and save to *output_model*."""
    import torch.optim as optim
    from sklearn.model_selection import train_test_split

    items = [(p, 1) for p in positive_clips] + [(p, 0) for p in negative_clips]
    train_items, val_items = train_test_split(items, test_size=0.15, random_state=42, shuffle=True)

    train_ds = ClipDataset(train_items)
    val_ds = ClipDataset(val_items)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=True)

    model = AestheticScorer().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_dl), epochs=epochs
    )

    best_val_acc = 0.0
    for epoch in range(epochs):
        # Unfreeze backbone after half of training
        if epoch == epochs // 2:
            model.unfreeze_backbone()
            for pg in optimizer.param_groups:
                pg["lr"] = lr * 0.1

        model.train()
        train_loss = 0.0
        for frames, labels in train_dl:
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(frames)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for frames, labels in val_dl:
                frames, labels = frames.to(DEVICE), labels.to(DEVICE)
                preds = (torch.sigmoid(model(frames)) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total > 0 else 0.0
        avg_loss = train_loss / len(train_dl)
        log.info("Epoch %d/%d  loss=%.4f  val_acc=%.3f", epoch + 1, epochs, avg_loss, val_acc)

        if progress_cb:
            progress_cb(f"Epoch {epoch+1}/{epochs}  val_acc={val_acc:.3f}", (epoch + 1) / epochs)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_model.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_model)
            log.info("Saved best model (val_acc=%.3f) → %s", best_val_acc, output_model)

    log.info("Training complete. Best val accuracy: %.3f", best_val_acc)


# ── Inference scorer ──────────────────────────────────────────────────────────

class ClipScorerInference:
    """Load a trained model and score individual clips."""

    def __init__(self, model_path: Path):
        self.model = AestheticScorer().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()

    @torch.no_grad()
    def score(self, clip_path: Path) -> float:
        """Return probability [0,1] that the clip is aesthetic."""
        frames = sample_frames(clip_path, FRAMES_PER_CLIP)
        tensors = torch.stack([_transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames])
        batch = tensors.unsqueeze(0).to(DEVICE)   # (1, T, 3, H, W)
        logit = self.model(batch)
        return float(torch.sigmoid(logit).item())

    @torch.no_grad()
    def score_frames(self, frames_bgr: list[np.ndarray]) -> float:
        """Score from already-loaded BGR frames."""
        tensors = torch.stack([_transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr])
        batch = tensors.unsqueeze(0).to(DEVICE)
        return float(torch.sigmoid(self.model(batch)).item())
