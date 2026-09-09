import pytest

from astra_yam import astra_client


def test_key_from_secrets_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(astra_client, "SECRET_DIRS", [tmp_path / "nope", tmp_path])
    monkeypatch.setattr(astra_client, "REPO_ROOT", tmp_path)   # no .env there
    (tmp_path / "OPENAI_API_KEY").write_text("sk-test-123\n")
    key, source = astra_client.find_api_key("OPENAI_API_KEY")
    assert key == "sk-test-123" and source.endswith("OPENAI_API_KEY")
    assert astra_client.load_api_key("OPENAI_API_KEY") == "sk-test-123"


def test_key_file_export_form_and_env_priority(tmp_path, monkeypatch):
    monkeypatch.setattr(astra_client, "SECRET_DIRS", [tmp_path])
    monkeypatch.setattr(astra_client, "REPO_ROOT", tmp_path)
    (tmp_path / "OPENAI_API_KEY").write_text('export OPENAI_API_KEY="sk-from-file"\n')
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert astra_client.find_api_key("OPENAI_API_KEY")[0] == "sk-from-file"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    assert astra_client.find_api_key("OPENAI_API_KEY") == ("sk-from-env", "environment")


def test_missing_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(astra_client, "SECRET_DIRS", [tmp_path])
    monkeypatch.setattr(astra_client, "REPO_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="not found"):
        astra_client.load_api_key("OPENAI_API_KEY")
