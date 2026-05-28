"""Retry and concurrency helpers for LLM-heavy KG steps."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Generic, Iterable, List, Optional, TypeVar

from tqdm import tqdm

from . import logger

T = TypeVar("T")
R = TypeVar("R")


class SkipItem(Exception):
    """Raised when the user explicitly skips the current item."""


def retry_or_skip(label: str, fn: Callable[[], R], initial_error: Optional[Exception] = None) -> R:
    """Retry the same item until it succeeds, or skip only when the user types n."""
    error = initial_error
    while True:
        if error is not None:
            logger.error(f"{label} 自动重试后仍失败: {type(error).__name__}: {error}")
            try:
                choice = input("\n>>> 回车重试当前项；输入 n 跳过继续: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                raise KeyboardInterrupt from error
            if choice == "n":
                raise SkipItem(str(error)) from error
            error = None
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            error = exc


def parallel_map_retryable(
    items: Iterable[T],
    worker: Callable[[T], R],
    retry_worker: Callable[[T], R],
    labeler: Callable[[T], str],
    *,
    concurrency: int,
    desc: str,
    unit: str,
) -> List[Optional[R]]:
    """Run independent work in parallel; failed items are retried/skipped on the main thread."""
    item_list = list(items)
    results: List[Optional[R]] = [None] * len(item_list)
    max_workers = max(1, concurrency)

    if max_workers == 1:
        for idx, item in enumerate(tqdm(item_list, desc=desc, unit=unit)):
            try:
                results[idx] = retry_or_skip(labeler(item), lambda item=item: retry_worker(item))
            except SkipItem:
                logger.warn(f"已跳过: {labeler(item)}")
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(worker, item): (idx, item)
            for idx, item in enumerate(item_list)
        }
        for future in tqdm(as_completed(future_map), total=len(future_map), desc=desc, unit=unit):
            idx, item = future_map[future]
            try:
                results[idx] = future.result()
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                try:
                    results[idx] = retry_or_skip(
                        labeler(item),
                        lambda item=item: retry_worker(item),
                        initial_error=exc,
                    )
                except SkipItem:
                    logger.warn(f"已跳过: {labeler(item)}")
    return results
