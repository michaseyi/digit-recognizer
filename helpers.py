import asyncio
from random import random
from typing import Callable, Dict, Generator


def id_generator() -> Generator[int, None, None]:
    i: int = 0
    while True:
        yield i
        i += 1


get_id = id_generator()
tasks: Dict[int, asyncio.Task] = {}


def set_timeout(callback: Callable, duration: float) -> int:
    async def routine():
        await asyncio.sleep(duration)
        callback()

    task = asyncio.create_task(routine())
    random_id = get_id.__next__()
    tasks[random_id] = task
    return random_id


def set_interval(callback: Callable, duration: float) -> int:
    async def routine():
        while True:
            await asyncio.sleep(duration)
            callback()

    task = asyncio.create_task(routine())
    random_id = get_id.__next__()
    tasks[random_id] = task
    return random_id


def clear_timeout(id: int) -> None:
    tasks[id].cancel()


def clear_interval(id: int) -> None:
    clear_timeout(id)
