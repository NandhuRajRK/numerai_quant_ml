from __future__ import annotations

import argparse

from numerai_quant.data import resolve_download_plan


def test_default_download_plan_fetches_train_only() -> None:
    args = argparse.Namespace(
        all=False,
        train_only=False,
        validation_only=False,
        live_only=False,
        parallel_secondary=False,
    )

    plan = resolve_download_plan(args)

    assert plan.primary_file_keys == ["train_file"]
    assert plan.secondary_file_keys == []
    assert plan.parallel_secondary is False


def test_all_download_plan_defers_secondary_files() -> None:
    args = argparse.Namespace(
        all=True,
        train_only=False,
        validation_only=False,
        live_only=False,
        parallel_secondary=True,
    )

    plan = resolve_download_plan(args)

    assert plan.primary_file_keys == ["train_file"]
    assert plan.secondary_file_keys == ["validation_file", "live_file"]
    assert plan.parallel_secondary is True
