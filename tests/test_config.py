from pathlib import Path

from core.config import load_settings


def test_load_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
rag:
  system_prompt: "Use context only."
paths:
  raw_data_dir: "{raw}"
  processed_data_dir: "{processed}"
  index_dir: "{index}"
  cache_dir: "{cache}"
vectorstore:
  persist_directory: "{chroma}"
conversation:
  long_term_memory_path: "{memory}"
evaluation:
  output_path: "{eval_out}"
""".format(
            raw=tmp_path / "raw",
            processed=tmp_path / "processed",
            index=tmp_path / "index",
            cache=tmp_path / "cache",
            chroma=tmp_path / "chroma",
            memory=tmp_path / "cache" / "memory.json",
            eval_out=tmp_path / "processed" / "eval.jsonl",
        ),
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.rag.system_prompt == "Use context only."
    assert settings.paths.raw_data_dir.exists()
    assert settings.vectorstore.persist_directory.exists()
