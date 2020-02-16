from concurrent.futures import Future
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List


def parallelize_task(num_workers, iterator, task, **task_kwargs):
    chunk_size = len(iterator) // num_workers
    print("chunk_size", chunk_size)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        tasks: List[Future] = []
        start_index = 0
        for x in range(num_workers):
            end_index = min(start_index + chunk_size + 1, len(iterator))
            chunk = iterator[start_index:end_index]
            tasks.append(executor.submit(task, chunk, **task_kwargs))
            start_index = end_index

        return [task.result() for task in tasks]
