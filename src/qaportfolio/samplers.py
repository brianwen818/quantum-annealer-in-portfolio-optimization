"""Sampler 介面：預設使用 D-Wave Ocean 的古典模擬退火（neal），
設定環境變數 DWAVE_API_TOKEN 後可切換到真實 QPU 或 Leap hybrid solver。

本研究所有結果皆以 `neal` 產生，並未使用量子硬體。

注意：50 檔 × 7 bits 的 QUBO 是全連接（dense）問題，約 350 個變數，
超過 Advantage QPU 能直接嵌入的完全圖大小（約 177），
因此實際送 QPU 時需減少資產數／精度，或改用 hybrid solver。
"""
from __future__ import annotations

import os

import dimod


def sample_bqm(bqm: dimod.BinaryQuadraticModel, kind: str, params: dict, seed: int) -> dimod.SampleSet:
    if kind == "neal":
        import neal

        return neal.SimulatedAnnealingSampler().sample(
            bqm, num_reads=params["num_reads"], num_sweeps=params["num_sweeps"], seed=seed)

    if not os.environ.get("DWAVE_API_TOKEN"):
        raise RuntimeError(f"sampler='{kind}' 需要設定環境變數 DWAVE_API_TOKEN")

    if kind == "qpu":
        from dwave.system import DWaveSampler, EmbeddingComposite

        return EmbeddingComposite(DWaveSampler()).sample(
            bqm, num_reads=params.get("num_reads", 100), label="qaportfolio")
    if kind == "hybrid":
        from dwave.system import LeapHybridSampler

        return LeapHybridSampler().sample(bqm, time_limit=params.get("time_limit", 5),
                                          label="qaportfolio")
    raise ValueError(f"unknown sampler: {kind}")
