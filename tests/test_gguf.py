from click.testing import CliRunner
from llm.cli import cli
import json
import pathlib

TEST_USER_DIR = pathlib.Path(__file__).parent.parent / "test-llm-user-dir"
TEST_USER_DIR.mkdir(exist_ok=True)

# This model is just 30MB
# Originally from:
# https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1/tree/main/gguf
EMBED_URL = (
    "https://raw.githubusercontent.com/simonw/test-files-for-llm-gguf/refs/"
    "heads/main/mxbai-embed-xsmall-v1-q8_0.gguf"
)


def test_embed_with_tiny_model(monkeypatch):
    monkeypatch.setenv("LLM_USER_PATH", str(TEST_USER_DIR))
    runner = CliRunner(mix_stderr=False)
    model_path = TEST_USER_DIR / "gguf" / "models" / "mxbai-embed-xsmall-v1-q8_0.gguf"
    if not model_path.exists():
        result = runner.invoke(cli, ["gguf", "download-embed-model", EMBED_URL])
        assert result.exit_code == 0
        assert "Download" in result.output
    # Model should exist
    assert model_path.exists()
    # Try to embed "hello world"
    result2 = runner.invoke(
        cli, ["embed", "-c", "hello world", "-m", "gguf/mxbai-embed-xsmall-v1-q8_0"]
    )
    assert result2.exit_code == 0, result2.output
    try:
        vector = json.loads(result2.output)
    except json.JSONDecodeError:
        assert False, result2.output
    assert len(vector) == 384


def test_register_and_unregister_model(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    runner = CliRunner()
    # register-model needs a file that exists; contents are irrelevant here
    dummy = tmp_path / "fake.gguf"
    dummy.write_bytes(b"not a real model")
    models_file = tmp_path / "gguf" / "models.json"

    result = runner.invoke(cli, ["gguf", "register-model", "fake-model", str(dummy)])
    assert result.exit_code == 0, result.output
    assert "fake-model" in json.loads(models_file.read_text())

    result2 = runner.invoke(cli, ["gguf", "unregister-model", "fake-model"])
    assert result2.exit_code == 0, result2.output
    assert "fake-model" not in json.loads(models_file.read_text())

    # Unregistering again should fail cleanly with a non-zero exit code
    result3 = runner.invoke(cli, ["gguf", "unregister-model", "fake-model"])
    assert result3.exit_code != 0


def test_register_and_unregister_embed_model(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    runner = CliRunner()
    dummy = tmp_path / "fake-embed.gguf"
    dummy.write_bytes(b"not a real model")
    models_file = tmp_path / "gguf" / "embed-models.json"

    result = runner.invoke(
        cli, ["gguf", "register-embed-model", "fake-embed", str(dummy)]
    )
    assert result.exit_code == 0, result.output
    assert "fake-embed" in json.loads(models_file.read_text())

    result2 = runner.invoke(cli, ["gguf", "unregister-embed-model", "fake-embed"])
    assert result2.exit_code == 0, result2.output
    assert "fake-embed" not in json.loads(models_file.read_text())

    result3 = runner.invoke(cli, ["gguf", "unregister-embed-model", "fake-embed"])
    assert result3.exit_code != 0
