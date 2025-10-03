from __future__ import annotations

import math
from typing import Callable, Iterable

from colorama import Fore, init
import numpy as np


DEFAULT_COINS: tuple[int, ...] = (50, 25, 10, 5, 2, 1)


def _prepare_coins(coins: Iterable[int] | None) -> tuple[int, ...]:
    prepared = tuple(sorted(set(DEFAULT_COINS if coins is None else coins), reverse=True))
    if not prepared:
        raise ValueError("Список номіналів монет не може бути порожнім")
    if any(c <= 0 for c in prepared):
        raise ValueError("Номінали монет повинні бути додатніми цілими числами")
    return prepared


def find_coins_greedy(amount: int, coins: Iterable[int] | None = None) -> dict[int, int]:
    """Повертає решту за жадібною стратегією {номінал: кількість}."""

    if amount < 0:
        raise ValueError("Сума повинна бути невідʼємною")

    denominations = _prepare_coins(coins)

    result: dict[int, int] = {}
    remaining = amount
    for coin in denominations:
        if remaining <= 0:
            break
        quotient, remaining = divmod(remaining, coin)
        if quotient:
            result[coin] = quotient
    return result


def find_min_coins(amount: int, coins: Iterable[int] | None = None) -> dict[int, int]:
    """Повертає оптимальний набір монет за допомогою динамічного програмування {номінал: кількість}."""

    if amount < 0:
        raise ValueError("Сума повинна бути невідʼємною")
    if amount == 0:
        return {}

    denominations = tuple(sorted(_prepare_coins(coins)))

    dp = [0] + [math.inf] * amount
    last = [-1] * (amount + 1)

    for value in range(1, amount + 1):
        best = math.inf
        best_coin = -1
        for coin in denominations:
            if coin > value:
                break
            candidate = dp[value - coin] + 1
            if candidate < best:
                best = candidate
                best_coin = coin
        dp[value] = best
        last[value] = best_coin

    if not math.isfinite(dp[amount]):
        raise ValueError("Неможливо зібрати задану суму з наданих монет")

    result: dict[int, int] = {}
    value = amount
    while value:
        coin = last[value]
        if coin == -1:
            raise RuntimeError("Помилка відновлення рішення")
        result[coin] = result.get(coin, 0) + 1
        value -= coin

    return dict(sorted(result.items()))


def monte_carlo_integration(
    func: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    num_samples: int = 100_000,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Наближено обчислює ∫_a^b func(x) dx та повертає (оцінку, стандартну похибку)."""

    if b <= a:
        raise ValueError("Верхня межа повинна бути більшою за нижню")
    if num_samples <= 0:
        raise ValueError("Кількість вибірок повинна бути додатньою")

    generator = rng or np.random.default_rng()
    samples = generator.uniform(a, b, num_samples)
    values = func(samples)

    mean_value = float(np.mean(values))
    integral_estimate = (b - a) * mean_value

    if num_samples == 1:
        return integral_estimate, 0.0

    variance = float(np.var(values, ddof=1))
    standard_error = (b - a) * math.sqrt(variance / num_samples)
    return integral_estimate, standard_error


init(autoreset=True)


def print_task_header(task_number: int) -> None:
    print()
    print(Fore.GREEN + f"--- Завдання {task_number} ---")
