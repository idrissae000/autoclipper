"""Tests for FaceDatabase persistence and lookup (no GPU required)."""

import pickle
from pathlib import Path

import numpy as np
import pytest

from autoclipper.face_recognition import FaceDatabase, SIMILARITY_THRESHOLD


def _random_unit(dim=512):
    v = np.random.default_rng(0).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_face_database_find_closest():
    db = FaceDatabase()
    emb_alice = _random_unit()
    emb_bob = _random_unit()

    db._entries["alice"] = emb_alice
    db._entries["bob"] = emb_bob

    # Should match alice when we query with alice's embedding
    name, dist = db.find(emb_alice)
    assert name == "alice"
    assert dist < 1e-5


def test_face_database_empty_returns_none():
    db = FaceDatabase()
    name, dist = db.find(_random_unit())
    assert name is None
    assert dist == 1.0


def test_face_database_save_load(tmp_path):
    db = FaceDatabase()
    db._entries["char1"] = _random_unit()

    pkl = tmp_path / "db.pkl"
    db.save(pkl)

    db2 = FaceDatabase()
    db2.load(pkl)
    assert "char1" in db2._entries
    np.testing.assert_allclose(db._entries["char1"], db2._entries["char1"])


def test_face_database_has():
    db = FaceDatabase()
    assert not db.has("nobody")
    db._entries["someone"] = _random_unit()
    assert db.has("someone")
