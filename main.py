from __future__ import annotations

import time
from typing import Iterable, Sequence

import numpy as np
import scipy.integrate as spi
from colorama import Fore, init

from lib import DEFAULT_COINS, find_coins_greedy, find_min_coins, monte_carlo_integration


init(autoreset=True)


def _format_change(change: dict[int, int]) -> str:
    return " ".join(f"{coin}:{count}" for coin, count in change.items()) or "-"


def benchmark_coin_algorithms(amounts: Iterable[int], coins: Sequence[int]) -> None:
    print("\nЗавдання 1 — Бенчмарки розміну монет")

    results: list[tuple[int, dict[int, int], dict[int, int], int, int]] = []
    for amount in amounts:
        start_ns = time.perf_counter_ns()
        greedy_result = find_coins_greedy(amount, coins)
        greedy_elapsed = time.perf_counter_ns() - start_ns

        start_ns = time.perf_counter_ns()
        dp_result = find_min_coins(amount, coins)
        dp_elapsed = time.perf_counter_ns() - start_ns

        results.append((amount, greedy_result, dp_result, greedy_elapsed, dp_elapsed))

    for amount, greedy_result, dp_result, greedy_ns, dp_ns in results:
        greedy_ms = greedy_ns / 1_000_000
        dp_ms = dp_ns / 1_000_000
        speedup = dp_ms / greedy_ms if greedy_ms else float("inf")
        mismatch = greedy_result != dp_result

        greedy_label = (
            f"жадібний {greedy_ms*1_000:.2f} µs"
            if greedy_ms < 1
            else f"жадібний {greedy_ms:.3f} ms"
        )
        dp_label = (
            f"ДП {dp_ms*1_000:.2f} µs" if dp_ms < 1 else f"ДП {dp_ms:.3f} ms"
        )

        print(f"Сума {amount:>6} → {greedy_label} :: {_format_change(greedy_result)}")
        print(
            f"{'':>12}   {dp_label} :: {_format_change(dp_result)}"
            f" (прискорення ×{speedup:,.1f})"
        )
        if mismatch:
            print(
                Fore.YELLOW
                + " " * 12
                + "Попередження: результати відрізняються. ДП надає оптимальну комбінацію."
            )
        print()


def run_monte_carlo_demo() -> None:
    print("\nЗавдання 2 — Інтегрування методом Монте-Карло")

    def func(x: np.ndarray) -> np.ndarray:
        return x**2

    a, b = 0.0, 2.0
    samples = 200_000
    rng = np.random.default_rng(seed=42)

    mc_estimate, mc_std_err = monte_carlo_integration(func, a, b, samples, rng=rng)
    analytical_solution = (b**3 - a**3) / 3
    quad_result, quad_error = spi.quad(lambda x: x**2, a, b)

    print("Функція: f(x) = x^2")
    print(f"Інтервал: [{a:g}, {b:g}]")
    print(
        "Оцінка Монте-Карло: "
        f"{mc_estimate:.8f} (±{mc_std_err:.8f}, {samples:,} вибірок)"
    )
    print(f"Аналітичне розвʼязання: {analytical_solution:.8f}")
    print(f"Результат quad: {quad_result:.8f} (±{quad_error:.2e})")
    print(
        "Відмінності — Монте-Карло vs quad: "
        f"{abs(mc_estimate - quad_result):.8f};"
        f" Монте-Карло vs аналітичне: {abs(mc_estimate - analytical_solution):.8f}"
    )


def main() -> None:
    test_amounts = (
        1,
        7,
        23,
        99,
        123,
        512,
        1_000,
        5_000,
        10_000,
        50_000,
        100_000,
        200_000,
    )

    benchmark_coin_algorithms(test_amounts, DEFAULT_COINS)
    run_monte_carlo_demo()


if __name__ == "__main__":
    main()
