"""
infrastructure/brightway/montecarlo/_distribution_strategies:
Estrategias de muestreo por distribución estadística.

Cada estrategia encapsula la lógica de muestreo de una distribución específica,
permitiendo agregar nuevas distribuciones sin modificar el código existente (OCP).

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SamplingStrategy(ABC):
    """Interfaz abstracta para estrategias de muestreo."""

    @abstractmethod
    def sample_scalar(
        self, params: dict[str, Any], nominal: float, rng: np.random.Generator
    ) -> float:
        """Genera una muestra escalar."""
        ...

    @abstractmethod
    def sample_vectorized(
        self, params: dict[str, Any], nominal: float, rng: np.random.Generator, n: int
    ) -> np.ndarray:
        """Genera un vector de n muestras."""
        ...


class DeterministicStrategy(SamplingStrategy):
    """Sin incertidumbre: retorna el valor nominal."""

    def sample_scalar(
        self, params: dict[str, Any], nominal: float, rng: np.random.Generator
    ) -> float:
        return float(nominal)

    def sample_vectorized(
        self, params: dict[str, Any], nominal: float, rng: np.random.Generator, n: int
    ) -> np.ndarray:
        return np.full(n, nominal, dtype=np.float64)


class LognormalStrategy(SamplingStrategy):
    """Distribución lognormal: asimétrica positiva."""

    def sample_scalar(
        self, params: dict[str, Any], nominal: float, rng: np.random.Generator
    ) -> float:
        loc = float(params.get("loc", np.log(abs(nominal)) if nominal > 0 else 0.0))
        scale = float(params.get("scale", 0.1))
        val = rng.lognormal(mean=loc, sigma=scale)
        return float(val if nominal >= 0 else -val)

    def sample_vectorized(
        self, params: dict[str, Any], nominal: float, rng: np.random.Generator, n: int
    ) -> np.ndarray:
        loc = float(params.get("loc", np.log(abs(nominal)) if nominal > 0 else 0.0))
        scale = float(params.get("scale", 0.1))
        vals = rng.lognormal(mean=loc, sigma=scale, size=n)
        return vals if nominal >= 0 else -vals


class NormalStrategy(SamplingStrategy):
    """Distribución normal: simétrica."""

    def sample_scalar(
        self, params: dict[str, Any], nominal: float, rng: np.random.Generator
    ) -> float:
        loc = float(params.get("loc", nominal))
        scale = float(params.get("scale", abs(nominal) * 0.1))
        return float(rng.normal(loc=loc, scale=max(abs(scale), 1e-12)))

    def sample_vectorized(
        self, params: dict[str, Any], nominal: float, rng: np.random.Generator, n: int
    ) -> np.ndarray:
        loc = float(params.get("loc", nominal))
        scale = float(params.get("scale", abs(nominal) * 0.1))
        return rng.normal(loc=loc, scale=max(abs(scale), 1e-12), size=n)


class UniformStrategy(SamplingStrategy):
    """Distribución uniforme: rango plano."""

    def sample_scalar(
        self, params: dict[str, Any], nominal: float, rng: np.random.Generator
    ) -> float:
        lo = float(params.get("minimum", nominal * 0.9))
        hi = float(params.get("maximum", nominal * 1.1))
        if lo >= hi:
            lo, hi = min(lo, hi) * 0.9, max(lo, hi) * 1.1
        return float(rng.uniform(lo, hi))

    def sample_vectorized(
        self, params: dict[str, Any], nominal: float, rng: np.random.Generator, n: int
    ) -> np.ndarray:
        lo = float(params.get("minimum", nominal * 0.9))
        hi = float(params.get("maximum", nominal * 1.1))
        if lo >= hi:
            lo, hi = min(lo, hi) * 0.9, max(lo, hi) * 1.1
        return rng.uniform(lo, hi, size=n)


class TriangularStrategy(SamplingStrategy):
    """Distribución triangular: mínimo, moda, máximo."""

    def sample_scalar(
        self, params: dict[str, Any], nominal: float, rng: np.random.Generator
    ) -> float:
        lo = float(params.get("minimum", nominal * 0.8))
        mode = float(params.get("loc", nominal))
        hi = float(params.get("maximum", nominal * 1.2))
        lo = min(lo, mode)
        hi = max(hi, mode)
        if lo == hi:
            return float(nominal)
        return float(rng.triangular(lo, mode, hi))

    def sample_vectorized(
        self, params: dict[str, Any], nominal: float, rng: np.random.Generator, n: int
    ) -> np.ndarray:
        lo = float(params.get("minimum", nominal * 0.8))
        mode = float(params.get("loc", nominal))
        hi = float(params.get("maximum", nominal * 1.2))
        lo = min(lo, mode)
        hi = max(hi, mode)
        if lo == hi:
            return np.full(n, nominal, dtype=np.float64)
        return rng.triangular(lo, mode, hi, size=n)


class WeibullStrategy(SamplingStrategy):
    """Distribución Weibull: procesos con falla o degradación."""

    def sample_scalar(
        self, params: dict[str, Any], nominal: float, rng: np.random.Generator
    ) -> float:
        shape = float(params.get("shape", 1.0))
        scale = float(params.get("scale", nominal))
        return float(rng.weibull(a=shape) * scale)

    def sample_vectorized(
        self, params: dict[str, Any], nominal: float, rng: np.random.Generator, n: int
    ) -> np.ndarray:
        shape = float(params.get("shape", 1.0))
        scale = float(params.get("scale", nominal))
        return rng.weibull(a=shape, size=n) * scale
