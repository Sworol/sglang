import importlib.util
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ModuleNotFoundError:
    # The CI registry is parsed from the source by AST. This fallback lets the
    # focused unit test run in minimal local environments without importing the
    # full sglang package.
    def register_cpu_ci(*args, **kwargs):
        return None


try:
    from sglang.test.test_utils import CustomTestCase
except ModuleNotFoundError:
    CustomTestCase = unittest.TestCase


register_cpu_ci(est_time=3, suite="base-a-test-cpu")


_REPO_ROOT = Path(__file__).resolve().parents[4]
_NVTX_UTILS_PATH = _REPO_ROOT / "python/sglang/srt/utils/nvtx_utils.py"


class _EnvFlag:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _RecordingContext:
    def __init__(self, events, label):
        self.events = events
        self.label = label

    def __enter__(self):
        self.events.append(f"{self.label}:enter")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.events.append(f"{self.label}:exit")


def _fake_torch():
    torch_mod = ModuleType("torch")
    torch_mod.autograd = SimpleNamespace(_profiler_enabled=Mock(return_value=False))
    torch_mod.profiler = SimpleNamespace(record_function=Mock())
    return torch_mod


def _fake_environ(*, scheduler_nvtx=False, operations_nvtx=False):
    environ_mod = ModuleType("sglang.srt.environ")
    environ_mod.envs = SimpleNamespace(
        SGLANG_ENABLE_NVTX_SCHEDULER=_EnvFlag(scheduler_nvtx),
        SGLANG_ENABLE_NVTX_OPERATIONS=_EnvFlag(operations_nvtx),
    )
    return environ_mod


def _fake_package(name):
    package = ModuleType(name)
    package.__path__ = []
    return package


def _load_nvtx_utils(
    *, scheduler_nvtx=False, operations_nvtx=False, nvtx_available=False
):
    module_name = (
        f"_test_nvtx_utils_{scheduler_nvtx}_{operations_nvtx}_{nvtx_available}"
    )
    spec = importlib.util.spec_from_file_location(module_name, _NVTX_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)

    modules = {
        "torch": _fake_torch(),
        "sglang": _fake_package("sglang"),
        "sglang.srt": _fake_package("sglang.srt"),
        "sglang.srt.environ": _fake_environ(
            scheduler_nvtx=scheduler_nvtx,
            operations_nvtx=operations_nvtx,
        ),
    }
    if nvtx_available:
        nvtx_module = ModuleType("nvtx")
        nvtx_module.annotate = Mock(return_value=_RecordingContext([], "nvtx"))
        modules["nvtx"] = nvtx_module

    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class TestNvtxUtils(CustomTestCase):
    def setUp(self):
        self.nvtx_utils = _load_nvtx_utils()

    def test_profile_range_returns_noop_when_no_profiler_or_nvtx(self):
        nvtx_utils = self.nvtx_utils

        with patch.object(
            nvtx_utils.torch.autograd, "_profiler_enabled", return_value=False
        ):
            context = nvtx_utils.profile_range("scheduler.run_batch")

        self.assertIs(context, nvtx_utils._NULL_CONTEXT)
        with context:
            pass

    def test_profile_range_records_torch_and_nvtx_contexts(self):
        nvtx_utils = self.nvtx_utils
        events = []

        def record_function(name):
            return _RecordingContext(events, f"record:{name}")

        def annotate(name, color=None):
            return _RecordingContext(events, f"nvtx:{name}:{color}")

        with (
            patch.object(
                nvtx_utils.torch.autograd, "_profiler_enabled", return_value=True
            ),
            patch.object(nvtx_utils.torch.profiler, "record_function", record_function),
            patch.object(
                nvtx_utils,
                "_nvtx_module",
                SimpleNamespace(annotate=annotate),
            ),
        ):
            with nvtx_utils.profile_range("scheduler.run_batch", nvtx_enabled=True):
                events.append("body")

        self.assertEqual(
            events,
            [
                "record:scheduler.run_batch:enter",
                "nvtx:scheduler.run_batch:red:enter",
                "body",
                "nvtx:scheduler.run_batch:red:exit",
                "record:scheduler.run_batch:exit",
            ],
        )

    def test_profile_range_uses_explicit_nvtx_color(self):
        nvtx_utils = self.nvtx_utils
        annotate = Mock(return_value=_RecordingContext([], "nvtx"))

        with (
            patch.object(
                nvtx_utils.torch.autograd, "_profiler_enabled", return_value=False
            ),
            patch.object(
                nvtx_utils, "_nvtx_module", SimpleNamespace(annotate=annotate)
            ),
        ):
            with nvtx_utils.profile_range(
                "scheduler.run_batch", color="orange", nvtx_enabled=True
            ):
                pass

        annotate.assert_called_once_with("scheduler.run_batch", color="orange")

    def test_profile_method_wraps_function_and_preserves_metadata(self):
        nvtx_utils = self.nvtx_utils
        calls = []

        @contextmanager
        def fake_profile_range(debug_name, *, color=None, nvtx_enabled=False):
            calls.append((debug_name, color, nvtx_enabled, "enter"))
            try:
                yield
            finally:
                calls.append((debug_name, color, nvtx_enabled, "exit"))

        def add(left, right):
            """sample doc"""
            return left + right

        wrapped = nvtx_utils.profile_method(
            "custom.method", color="green", nvtx_enabled=True
        )(add)

        with patch.object(nvtx_utils, "profile_range", fake_profile_range):
            result = wrapped(2, 3)

        self.assertEqual(result, 5)
        self.assertEqual(wrapped.__name__, "add")
        self.assertEqual(wrapped.__doc__, "sample doc")
        self.assertEqual(
            calls,
            [
                ("custom.method", "green", True, "enter"),
                ("custom.method", "green", True, "exit"),
            ],
        )

    def test_prebound_helpers_keep_import_time_nvtx_flags(self):
        nvtx_utils = _load_nvtx_utils(
            scheduler_nvtx=True, operations_nvtx=True, nvtx_available=True
        )

        self.assertTrue(nvtx_utils.NVTX_AVAILABLE)
        self.assertTrue(nvtx_utils.NVTX_SCHEDULER_ENABLED)
        self.assertTrue(nvtx_utils.NVTX_OPERATIONS_ENABLED)
        self.assertIs(nvtx_utils.scheduler_nvtx_method.func, nvtx_utils.profile_method)
        self.assertEqual(
            nvtx_utils.scheduler_nvtx_method.keywords,
            {"nvtx_enabled": nvtx_utils.NVTX_SCHEDULER_ENABLED},
        )

        self.assertIs(nvtx_utils.operations_nvtx_range.func, nvtx_utils.profile_range)
        self.assertEqual(
            nvtx_utils.operations_nvtx_range.keywords,
            {"nvtx_enabled": nvtx_utils.NVTX_OPERATIONS_ENABLED},
        )


if __name__ == "__main__":
    unittest.main()
