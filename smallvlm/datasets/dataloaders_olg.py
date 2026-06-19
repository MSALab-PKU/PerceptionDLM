import time
from types import MethodType
import functools
import queue
import multiprocessing
import threading
import random

import torch
from torch.utils.data.dataloader import (DataLoader, _MultiProcessingDataLoaderIter, _utils,
                                         _sharding_worker_init_fn, _DatasetKind, IterDataPipe, MapDataPipe)


def apply_online_length_grouped_dataloader(dataloader: DataLoader):
    def _get_iterator(self):
        assert self.num_workers > 0
        self.check_worker_number_rationality()
        if not hasattr(self, '_iter_random_seed'):
            self._iter_random_seed = 67107
        else:
            self._iter_random_seed += 1
        return _OnlineLengthGroupedDataLoaderIter(self, self._iter_random_seed)

    if hasattr(dataloader, '_get_iterator'):
        dataloader._get_iterator = MethodType(_get_iterator, dataloader)
    else:
        raise AttributeError(f"{type(dataloader)} object has no attribute '_get_iterator'"
                             ", which is required to apply_online_length_grouped_dataloader.")


class _OnlineLengthGroupedDataLoaderIter(_MultiProcessingDataLoaderIter):
    def __init__(self, loader: DataLoader, random_seed: int):
        super(_MultiProcessingDataLoaderIter, self).__init__(loader)

        assert self._dataset_kind == _DatasetKind.Map
        assert not self._persistent_workers

        self._prefetch_factor = loader.prefetch_factor

        assert self._num_workers > 0
        assert self._prefetch_factor > 0

        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context

        self._worker_init_fn = loader.worker_init_fn

        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   Additional worker init function will take care of sharding in MP and Distributed
        if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
            self._worker_init_fn = functools.partial(
                _sharding_worker_init_fn, self._worker_init_fn, self._world_size, self._rank)

        # No certainty which module multiprocessing_context is
        self._worker_result_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
        self._worker_pids_set = False
        self._shutdown = False
        self._workers_done_event = multiprocessing_context.Event()

        self._index_queues = []
        self._workers = []
        for i in range(self._num_workers):
            # No certainty which module multiprocessing_context is
            index_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
            # Need to `cancel_join_thread` here!
            # See sections (2) and (3b) above.
            index_queue.cancel_join_thread()
            w = multiprocessing_context.Process(
                target=_utils.worker._worker_loop,
                args=(self._dataset_kind, self._dataset, index_queue,
                      self._worker_result_queue, self._workers_done_event,
                      self._auto_collation, null_collate_fn, self._drop_last,
                      self._base_seed, self._worker_init_fn, i, self._num_workers,
                      self._persistent_workers, self._shared_seed))
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self._workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)

        self._collate_queue = multiprocessing_context.Queue()
        buffer_size = min(max(int(0.9 * self._prefetch_factor), 2), self._prefetch_factor) * self._num_workers
        collate_process = multiprocessing_context.Process(
            target=_collate_loop,
            args=(self._worker_result_queue, self._collate_queue,
                  self._collate_fn, buffer_size, self._workers_done_event, random_seed))
        collate_process.daemon = True
        collate_process.start()
        self._collate_process = collate_process

        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()

            # Queue is not type-annotated
            self._data_queue = queue.Queue()  # type: ignore[var-annotated]
            if self._pin_memory_device == "xpu":
                current_device = torch.xpu.current_device()  # type: ignore[attr-defined]
            elif self._pin_memory_device == torch._C._get_privateuse1_backend_name():
                custom_device_mod = getattr(torch, torch._C._get_privateuse1_backend_name())
                current_device = custom_device_mod.current_device()
            else:
                current_device = torch.cuda.current_device()  # choose cuda for default
            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(self._collate_queue, self._data_queue,
                      current_device,
                      self._pin_memory_thread_done_event, self._pin_memory_device))
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            # Similar to workers (see comment above), we only register
            # pin_memory_thread once it is started.
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._collate_queue  # type: ignore[assignment]

        # In some rare cases, persistent workers (daemonic processes)
        # would be terminated before `__del__` of iterator is invoked
        # when main process exits
        # It would cause failure when pin_memory_thread tries to read
        # corrupted data from worker_result_queue
        # atexit is used to shutdown thread and child processes in the
        # right sequence before main process exits
        if self._persistent_workers and self._pin_memory:
            import atexit
            for w in self._workers:
                atexit.register(_MultiProcessingDataLoaderIter._clean_up_worker, w)

        # .pid can be None only before process is spawned (not the case, so ignore)
        _utils.signal_handling._set_worker_pids(id(self),
                                                tuple(w.pid for w in self._workers) + (self._collate_process.pid,))
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True
        self._reset(loader, first_iter=True)

    def _reset(self, loader, first_iter=False):
        self._collate_status = True
        self._end_of_data = False
        super()._reset(loader, first_iter)

    def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
        try:
            data = self._data_queue.get(timeout=timeout)
            return (True, data)
        except Exception as e:
            if self._collate_status and not self._collate_process.is_alive():
                raise RuntimeError(f'DataLoader collate process (pid {self._collate_process.pid}) exited unexpectedly')

            failed_workers = []
            for worker_id, w in enumerate(self._workers):
                if self._workers_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    self._mark_worker_as_unavailable(worker_id)
            if len(failed_workers) > 0:
                pids_str = ', '.join(str(w.pid) for w in failed_workers)
                raise RuntimeError(f'DataLoader worker (pid(s) {pids_str}) exited unexpectedly') from e
            if isinstance(e, queue.Empty):
                return (False, None)
            import tempfile
            import errno
            try:
                fds_limit_margin = 10
                fs = [tempfile.NamedTemporaryFile() for i in range(fds_limit_margin)]
            except OSError as e:
                if e.errno == errno.EMFILE:
                    raise RuntimeError(
                        "Too many open files. Communication with the"
                        " workers is no longer possible. Please increase the"
                        " limit using `ulimit -n` in the shell or change the"
                        " sharing strategy by calling"
                        " `torch.multiprocessing.set_sharing_strategy('file_system')`"
                        " at the beginning of your code") from None
            raise

    def _try_put_index(self):
        assert self._tasks_outstanding < self._prefetch_factor * self._num_workers

        try:
            index = self._next_index()
        except StopIteration:
            if not self._end_of_data:
                self._worker_result_queue.put((self._send_idx, None))
                self._end_of_data = True
            return
        for _ in range(self._num_workers):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                break
        else:
            # not found (i.e., didn't break)
            return

        self._index_queues[worker_queue_idx].put((self._send_idx, index))
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1

    def _mark_collate_as_unavailable(self, shutdown=False):
        assert self._collate_status or (self._persistent_workers and shutdown)
        self._collate_queue.put(None)
        self._collate_status = False
        assert self._workers_done_event.is_set() == shutdown

    def _shutdown_workers(self):
        if _utils is None or _utils.python_exit_status is True or _utils.python_exit_status is None:
            return

        if not self._shutdown:
            self._shutdown = True
            try:
                if hasattr(self, '_pin_memory_thread'):
                    self._pin_memory_thread_done_event.set()
                    self._collate_queue.put(None)
                    self._pin_memory_thread.join()
                    self._collate_queue.cancel_join_thread()
                    self._collate_queue.close()

                self._worker_result_queue.put((None, None))
                self._worker_result_queue.cancel_join_thread()
                self._worker_result_queue.close()

                # Exit workers now.
                self._workers_done_event.set()
                self._mark_collate_as_unavailable(shutdown=True)
                for worker_id in range(len(self._workers)):
                    if self._persistent_workers or self._workers_status[worker_id]:
                        self._mark_worker_as_unavailable(worker_id, shutdown=True)

                self._collate_process.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                for w in self._workers:
                    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)

                for q in self._index_queues:
                    q.cancel_join_thread()
                    q.close()
            finally:
                if self._worker_pids_set:
                    _utils.signal_handling._remove_worker_pids(id(self))
                    self._worker_pids_set = False
                if self._collate_process.is_alive():
                    self._collate_process.terminate()
                for w in self._workers:
                    if w.is_alive():
                        w.terminate()


def _collate_loop(in_queue, out_queue, collate_fn, buffer_size, done_event, random_seed=1042):
    torch.set_num_threads(1)
    rg = random.Random(random_seed)

    def collate_once(idx_buffer, data_buffer):
        if len(data_buffer) == 0:
            return
        idx_buffer = sorted(idx_buffer)
        data_buckets = bucketing_data(data_buffer)
        data_buffer.clear()
        rg.shuffle(data_buckets)
        assert len(idx_buffer) == len(data_buckets)
        while len(idx_buffer) > 0:
            idx, data = idx_buffer.pop(0), data_buckets.pop(0)
            data = collate_fn(data)
            while not done_event.is_set():
                try:
                    out_queue.put((idx, data), timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                    break
                except queue.Full:
                    continue

    cur_idx = 0
    idx_buffer = []
    data_buffer = []
    cache = {}
    _counter = 0.
    while not done_event.is_set():
        if cur_idx in cache:
            r = cache.pop(cur_idx)
        else:
            try:
                r = in_queue.get(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if r is None:
                break

        idx, data = r
        if idx > cur_idx:
            cache[idx] = r
            continue
        elif idx < cur_idx:
            raise ValueError(f"Unexpected idx {idx} < cur_idx {cur_idx} in collate loop.")

        if data is None:
            collate_once(idx_buffer, data_buffer)
            while not done_event.is_set():
                time.sleep(0.2)
            break
        cur_idx += 1
        if not isinstance(data, list):
            out_queue.put((idx, data), timeout=_utils.MP_STATUS_CHECK_INTERVAL)
            continue
        idx_buffer.append(idx)
        data_buffer.append(data)

        if len(idx_buffer) >= int(buffer_size * _counter):
            collate_once(idx_buffer, data_buffer)
            idx_buffer = []
            data_buffer = []
            _counter = min(1., _counter + 0.05)

    has_data = len(data_buffer) > 0 or len(cache) > 0
    idx_buffer.clear()
    data_buffer.clear()
    cache.clear()
    del idx_buffer
    del data_buffer
    del cache
    if has_data:
        raise RuntimeError("collate_loop exits with data.")


def bucketing_data(data_buffer):
    bucket_size = len(data_buffer[0])
    assert all(len(data) == bucket_size for data in data_buffer)

    data_buffer = sum(data_buffer, [])
    data_buffer = sorted(data_buffer, key=lambda data: data['input_ids'].size(1))
    data_buffer = [data_buffer[i:i+bucket_size] for i in range(0, len(data_buffer), bucket_size)]

    return data_buffer


def null_collate_fn(batch):
    return batch
