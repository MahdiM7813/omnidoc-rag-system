from modules.cache import DiskJsonCache


def test_disk_json_cache_round_trip(tmp_path) -> None:
    cache = DiskJsonCache(tmp_path, enabled=True, ttl_seconds=3600)

    cache.set("hello", {"value": 1})

    assert cache.get("hello") == {"value": 1}
    assert cache.get("missing") is None
