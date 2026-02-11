"""Tests for config.py — configuration loading and profile management."""

import os
import tempfile
from pathlib import Path
from unittest import mock

# Patch config paths before importing
_test_dir = tempfile.mkdtemp()
_test_config_dir = Path(_test_dir) / "config"
_test_data_dir = Path(_test_dir) / "data"
_test_config_dir.mkdir(parents=True, exist_ok=True)
_test_data_dir.mkdir(parents=True, exist_ok=True)


import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

with mock.patch("config.CONFIG_DIR", _test_config_dir), \
     mock.patch("config.DATA_DIR", _test_data_dir), \
     mock.patch("config.CONFIG_FILE", _test_config_dir / "config.yaml"), \
     mock.patch("config.SESSIONS_DIR", _test_data_dir / "sessions"):

    from config import Config, load_config, _apply_env_vars, _get_profile_from_yaml


class TestConfigDefaults:
    """Test default configuration values."""

    def test_default_config_values(self):
        c = Config()
        assert c.endpoint == "http://localhost:11434/v1"
        assert c.api_key == ""
        assert c.model == "llama3"
        assert c.temperature == 0.7
        assert c.max_tokens is None
        assert c.stream is True

    def test_display_key_empty(self):
        c = Config(api_key="")
        assert c.display_key() == "(none)"

    def test_display_key_short(self):
        c = Config(api_key="abc")
        assert c.display_key() == "****"

    def test_display_key_long(self):
        c = Config(api_key="sk-1234567890abcdef")
        assert c.display_key() == "sk-1…cdef"

    def test_display_endpoint_strips_trailing_slash(self):
        c = Config(endpoint="http://localhost:11434/v1/")
        assert c.display_endpoint() == "http://localhost:11434/v1"

    def test_to_dict_masks_key(self):
        c = Config(api_key="sk-1234567890abcdef")
        d = c.to_dict()
        assert d["api_key"] == "sk-1…cdef"
        assert d["model"] == "llama3"


class TestEnvVarOverrides:
    """Test environment variable override logic."""

    def test_string_overrides(self):
        config = {"endpoint": "old", "api_key": "old", "model": "old"}
        with mock.patch.dict(os.environ, {
            "CLIAI_ENDPOINT": "https://api.openai.com/v1",
            "CLIAI_API_KEY": "sk-test",
            "CLIAI_MODEL": "gpt-4",
        }):
            result = _apply_env_vars(config)
        assert result["endpoint"] == "https://api.openai.com/v1"
        assert result["api_key"] == "sk-test"
        assert result["model"] == "gpt-4"

    def test_float_override(self):
        config = {"temperature": 0.7}
        with mock.patch.dict(os.environ, {"CLIAI_TEMPERATURE": "0.3"}):
            result = _apply_env_vars(config)
        assert result["temperature"] == 0.3

    def test_int_override(self):
        config = {"max_tokens": None}
        with mock.patch.dict(os.environ, {"CLIAI_MAX_TOKENS": "1024"}):
            result = _apply_env_vars(config)
        assert result["max_tokens"] == 1024

    def test_bool_override(self):
        config = {"stream": True}
        with mock.patch.dict(os.environ, {"CLIAI_STREAM": "false"}):
            result = _apply_env_vars(config)
        assert result["stream"] is False

    def test_invalid_float_ignored(self):
        config = {"temperature": 0.7}
        with mock.patch.dict(os.environ, {"CLIAI_TEMPERATURE": "not_a_number"}):
            result = _apply_env_vars(config)
        assert result["temperature"] == 0.7


class TestProfileLoading:
    """Test YAML profile extraction."""

    def test_get_named_profile(self):
        yaml_data = {
            "profiles": {
                "openai": {"endpoint": "https://api.openai.com/v1", "model": "gpt-4"},
                "local": {"endpoint": "http://localhost:11434/v1", "model": "llama3"},
            }
        }
        profile = _get_profile_from_yaml(yaml_data, "openai")
        assert profile["endpoint"] == "https://api.openai.com/v1"
        assert profile["model"] == "gpt-4"

    def test_get_default_profile(self):
        yaml_data = {
            "default_profile": "local",
            "profiles": {
                "local": {"endpoint": "http://localhost:11434/v1"},
            }
        }
        profile = _get_profile_from_yaml(yaml_data, None)
        assert profile["endpoint"] == "http://localhost:11434/v1"

    def test_missing_profile_returns_empty(self, capsys):
        yaml_data = {"profiles": {"a": {"model": "x"}}}
        profile = _get_profile_from_yaml(yaml_data, "nonexistent")
        assert profile == {}

    def test_no_profiles_returns_empty(self):
        profile = _get_profile_from_yaml({}, None)
        assert profile == {}
