"""Required Windows CI entry point: unavailable prerequisites must fail, not skip."""
import os
import unittest


def main():
    if os.name != "nt":
        raise SystemExit("This required concurrency/reproduction suite must run on Windows")
    try:
        import lupa.lua54  # noqa: F401; fail instead of silently skipping mandatory CI.
    except ImportError as exc:
        raise SystemExit("Install test dependency lupa to execute real Lua I/O tests") from exc
    from test_status_io import StatusIOTests
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(StatusIOTests))
    if result.skipped:
        raise SystemExit("Windows Lua status I/O suite unexpectedly skipped tests")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
