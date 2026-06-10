#!/usr/bin/env python
"""Tests for unified Confluence credential resolution.

`resolve_confluence_credentials` combines non-secret config (from config.yaml)
with an env-only token, with env overrides for the non-secret fields. These
tests pass an explicit config dict and control the environment so they do not
depend on the package .env or a live Confluence.
"""

import pytest

from picasso_workflow.confluence import (
    resolve_confluence_credentials,
    NullConfluenceInterface,
)

_CONFIG = {
    "Confluence": {
        "URL": "https://conf.example/wiki",
        "Space": "OPS",
        "DefaultPage": "Ops Landing",
        "Username": "ops@example.com",
    },
    "ConfluenceTest": {
        "URL": "https://conf.example/wiki",
        "Space": "TESTSPACE",
        "DefaultPage": "Test Landing",
        "Username": "test@example.com",
    },
}

# every Confluence-related env var the resolver may read
_ALL_ENV = [
    "CONFLUENCE_URL",
    "CONFLUENCE_SPACE",
    "CONFLUENCE_BASE_PAGE",
    "CONFLUENCE_USERNAME",
    "CONFLUENCE_TOKEN",
    "CONFLUENCE_BEARER",
    "TEST_CONFLUENCE_URL",
    "TEST_CONFLUENCE_SPACE",
    "TEST_CONFLUENCE_PAGE",
    "TEST_CONFLUENCE_USERNAME",
    "TEST_CONFLUENCE_TOKEN",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ALL_ENV:
        monkeypatch.delenv(name, raising=False)


def test_non_secret_from_config_token_from_env(monkeypatch):
    monkeypatch.setenv("CONFLUENCE_TOKEN", "tok-123")
    creds = resolve_confluence_credentials("Confluence", config=_CONFIG)
    assert creds["base_url"] == "https://conf.example/wiki"
    assert creds["space_key"] == "OPS"
    assert creds["parent_page_title"] == "Ops Landing"
    assert creds["username"] == "ops@example.com"
    assert creds["token"] == "tok-123"


def test_token_is_env_only_never_from_config(monkeypatch):
    # a token accidentally placed in config must NOT be picked up
    cfg = {"Confluence": {**_CONFIG["Confluence"], "Token": "should-ignore"}}
    creds = resolve_confluence_credentials("Confluence", config=cfg)
    assert creds["token"] is None


def test_env_overrides_config_for_nonsecret(monkeypatch):
    monkeypatch.setenv("CONFLUENCE_SPACE", "OVERRIDE")
    monkeypatch.setenv("CONFLUENCE_TOKEN", "t")
    creds = resolve_confluence_credentials("Confluence", config=_CONFIG)
    assert creds["space_key"] == "OVERRIDE"  # env beats config
    assert creds["base_url"] == "https://conf.example/wiki"  # config kept


def test_bearer_alias_accepted(monkeypatch):
    monkeypatch.setenv("CONFLUENCE_BEARER", "legacy-tok")
    creds = resolve_confluence_credentials("Confluence", config=_CONFIG)
    assert creds["token"] == "legacy-tok"


def test_token_canonical_wins_over_alias(monkeypatch):
    monkeypatch.setenv("CONFLUENCE_TOKEN", "canon")
    monkeypatch.setenv("CONFLUENCE_BEARER", "legacy")
    creds = resolve_confluence_credentials("Confluence", config=_CONFIG)
    assert creds["token"] == "canon"


def test_surrounding_quotes_stripped(monkeypatch):
    # the CI runner wraps .env values in quotes
    monkeypatch.setenv("TEST_CONFLUENCE_TOKEN", '"quoted-tok"')
    monkeypatch.setenv("TEST_CONFLUENCE_SPACE", "'sp'")
    creds = resolve_confluence_credentials("ConfluenceTest", config=_CONFIG)
    assert creds["token"] == "quoted-tok"
    assert creds["space_key"] == "sp"


def test_test_profile_uses_test_section_and_token(monkeypatch):
    monkeypatch.setenv("TEST_CONFLUENCE_TOKEN", "test-tok")
    # an operational token must not leak into the test profile
    monkeypatch.setenv("CONFLUENCE_TOKEN", "ops-tok")
    creds = resolve_confluence_credentials("ConfluenceTest", config=_CONFIG)
    assert creds["space_key"] == "TESTSPACE"
    assert creds["token"] == "test-tok"


def test_missing_token_is_none_not_crash(monkeypatch):
    creds = resolve_confluence_credentials("ConfluenceTest", config=_CONFIG)
    assert creds["token"] is None  # graceful, enables skip


def test_missing_section_yields_none_fields(monkeypatch):
    creds = resolve_confluence_credentials("Confluence", config={})
    assert creds["base_url"] is None
    assert creds["token"] is None


def test_null_confluence_interface_inert():
    ci = NullConfluenceInterface()
    assert ci.create_page("p", "b") == "local"
    assert ci.get_page_properties("p") == ("local", "p")
    assert ci.update_page_content("p", "id", "body") is None
