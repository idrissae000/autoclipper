"""Tests for clip_scorer – model forward pass, frame sampling."""

import numpy as np
import pytest


def test_aesthetic_scorer_forward():
    torch = pytest.importorskip("torch")
    import torchvision.models as models
    from autoclipper.clip_scorer import AestheticScorer, FRAMES_PER_CLIP, EMBED_DIM

    # Initialise without downloading pretrained weights
    with torch.no_grad():
        model = AestheticScorer.__new__(AestheticScorer)
        torch.nn.Module.__init__(model)
        backbone = models.mobilenet_v3_large(weights=None)
        model.features = backbone.features
        model.avgpool = backbone.avgpool
        model.head = __import__("autoclipper.clip_scorer", fromlist=["AestheticHead"]).AestheticHead(EMBED_DIM)
    model.eval()

    batch = torch.randn(2, FRAMES_PER_CLIP, 3, 224, 224)
    with torch.no_grad():
        out = model(batch)

    assert out.shape == (2,)
    # logits should be finite
    assert torch.isfinite(out).all()


def test_sample_frames_returns_correct_count(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import cv2 as _cv2

    path = tmp_path / "test.mp4"
    fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
    w = _cv2.VideoWriter(str(path), fourcc, 24.0, (64, 64))
    for _ in range(24):
        w.write(np.zeros((64, 64, 3), dtype=np.uint8))
    w.release()

    from autoclipper.clip_scorer import sample_frames
    frames = sample_frames(path, n=8)
    assert len(frames) == 8
    assert all(f.shape == (64, 64, 3) for f in frames)


def test_aesthetic_head_binary_output():
    torch = pytest.importorskip("torch")
    from autoclipper.clip_scorer import AestheticHead

    head = AestheticHead()
    x = torch.randn(4, 960)
    out = head(x)
    assert out.shape == (4,)
    probs = torch.sigmoid(out)
    assert ((probs >= 0) & (probs <= 1)).all()
