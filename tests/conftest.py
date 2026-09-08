"""Shared pytest safety controls."""

import ipaddress
import socket
from collections.abc import Generator
from typing import Any

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live_sec: allows an explicitly requested live SEC test")
    config.addinivalue_line(
        "markers",
        "live_provider: allows explicitly selected live provider connectivity tests",
    )


def _is_local_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def block_external_network(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Prevent accidental external network access in deterministic tests."""

    if request.node.get_closest_marker("live_sec") or request.node.get_closest_marker(
        "live_provider"
    ):
        yield
        return

    original_getaddrinfo = socket.getaddrinfo
    original_connect = socket.socket.connect

    def guarded_getaddrinfo(host: str | bytes | None, *args: Any, **kwargs: Any) -> Any:
        decoded_host = host.decode() if isinstance(host, bytes) else host
        if decoded_host is not None and not _is_local_host(decoded_host):
            raise RuntimeError(f"external network disabled during tests: {decoded_host}")
        return original_getaddrinfo(host, *args, **kwargs)

    def guarded_connect(sock: socket.socket, address: Any) -> Any:
        if sock.family in {socket.AF_INET, socket.AF_INET6} and isinstance(address, tuple):
            host = str(address[0])
            if not _is_local_host(host):
                raise RuntimeError(f"external network disabled during tests: {host}")
        return original_connect(sock, address)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    yield
