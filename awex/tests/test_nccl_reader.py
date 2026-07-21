from awex.reader import nccl_reader


class _MetaServerClient:
    def __init__(self):
        self.get_calls = []
        self.deleted_keys = []

    def get_object(self, key, timeout=0, default_value=None):
        self.get_calls.append((key, timeout, default_value))
        return True

    def delete_if_exists(self, key):
        self.deleted_keys.append(key)


def test_wait_colocate_write_finished_only_rank0_deletes(monkeypatch):
    barrier_calls = []

    monkeypatch.setattr(nccl_reader.device_util, "current_device", lambda: 3)
    monkeypatch.setattr(
        nccl_reader.dist,
        "barrier",
        lambda group=None, device_ids=None: barrier_calls.append((group, device_ids)),
    )

    rank0_meta = _MetaServerClient()
    rank3_meta = _MetaServerClient()

    nccl_reader._wait_colocate_write_finished(rank0_meta, "write_finished_key", "pg", 0)
    nccl_reader._wait_colocate_write_finished(rank3_meta, "write_finished_key", "pg", 3)

    assert rank0_meta.get_calls == [("write_finished_key", 1024**3, None)]
    assert rank3_meta.get_calls == [("write_finished_key", 1024**3, None)]
    assert rank0_meta.deleted_keys == ["write_finished_key"]
    assert rank3_meta.deleted_keys == []
    assert barrier_calls == [
        ("pg", [3]),
        ("pg", [3]),
        ("pg", [3]),
        ("pg", [3]),
    ]
