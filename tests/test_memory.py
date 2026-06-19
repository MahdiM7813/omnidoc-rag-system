from modules.memory import LongTermMemoryStore, ShortTermMemory


def test_short_term_memory_window() -> None:
    memory = ShortTermMemory(max_turns=2)

    memory.add("q1", "a1")
    memory.add("q2", "a2")
    memory.add("q3", "a3")

    assert len(memory.turns) == 2
    assert memory.turns[0].user == "q2"


def test_long_term_memory_store(tmp_path) -> None:
    store = LongTermMemoryStore(tmp_path / "memory.json")
    memory = ShortTermMemory(max_turns=3)
    memory.add("q", "a")

    store.save_session("s1", memory)
    loaded = store.load_session("s1", max_turns=3)

    assert loaded.turns[0].assistant == "a"
