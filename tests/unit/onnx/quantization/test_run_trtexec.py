# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for ``_run_trtexec`` streaming / heartbeat UX."""

from __future__ import annotations

import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from modelopt.onnx.quantization import ort_utils


def _fake_popen(
    stdout_text: str = "", stderr_text: str = "", returncode: int = 0, hold_s: float = 0.0
):
    """Build a Popen-like object that emits the given stdout/stderr then exits."""

    class _Stream:
        def __init__(self, text: str):
            self._lines = text.splitlines(keepends=True)
            self._idx = 0

        def readline(self) -> str:
            if self._idx >= len(self._lines):
                return ""
            line = self._lines[self._idx]
            self._idx += 1
            return line

        def close(self) -> None:
            return None

    proc = MagicMock()
    proc.stdout = _Stream(stdout_text)
    proc.stderr = _Stream(stderr_text)
    proc.returncode = returncode
    start = time.monotonic()

    def _poll():
        if time.monotonic() - start < hold_s:
            return None
        return returncode

    def _wait(timeout=None):
        while _poll() is None:
            time.sleep(0.01)
        return returncode

    def _kill():
        nonlocal hold_s
        hold_s = 0.0

    proc.poll.side_effect = _poll
    proc.wait.side_effect = _wait
    proc.kill.side_effect = _kill
    return proc


def test_format_duration_units():
    assert ort_utils._format_duration(5) == "5s"
    assert ort_utils._format_duration(65) == "1m 5s"
    assert ort_utils._format_duration(3661) == "1h 1m 1s"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("[I] Building engine from model", True),
        ("[I] Engine built in 12.3 sec", True),
        ("[I] &&&& PASSED TensorRT.trtexec", True),
        ("[I] Some unrelated builder note", False),
        ("[V] Verbose tactic dump line", False),
    ],
)
def test_is_trtexec_progress_line(line, expected):
    assert ort_utils._is_trtexec_progress_line(line) is expected


def test_run_trtexec_streams_output_and_promotes_milestones(caplog):
    """Milestone lines are INFO; other lines stay DEBUG; full text is captured."""
    fake = _fake_popen(
        stdout_text=(
            "[I] Building engine from model\n"
            "[I] Some unrelated builder note\n"
            "[I] Engine built in 1.0 sec\n"
        ),
        stderr_text="warn-line\n",
        returncode=0,
    )

    with (
        caplog.at_level("DEBUG", logger="modelopt.onnx"),
        patch("subprocess.Popen", return_value=fake),
    ):
        result = ort_utils._run_trtexec(
            ["--onnx=model.onnx"], timeout=None, heartbeat_interval=60.0
        )

    assert result.returncode == 0
    assert "Building engine from model" in result.stdout
    assert "warn-line" in result.stderr
    messages = [r.getMessage() for r in caplog.records]
    assert any("Starting trtexec" in m for m in messages)
    assert any("[trtexec] [I] Building engine from model" in m for m in messages)
    assert any("trtexec finished" in m for m in messages)
    # Non-milestone line is debug-only, not promoted to a bare INFO "[trtexec] ..." line.
    assert not any(
        r.levelname == "INFO" and "Some unrelated builder note" in r.getMessage()
        for r in caplog.records
    )


def test_run_trtexec_emits_heartbeat_while_silent(caplog):
    """Heartbeats fire when the process is alive with no fresh output."""
    fake = _fake_popen(stdout_text="", hold_s=0.55)

    with (
        caplog.at_level("INFO", logger="modelopt.onnx"),
        patch("subprocess.Popen", return_value=fake),
    ):
        ort_utils._run_trtexec(timeout=None, heartbeat_interval=0.2)

    assert any("trtexec still running" in r.getMessage() for r in caplog.records)


def test_run_trtexec_skips_progress_noise_for_short_timeout(caplog):
    """Version-probe style calls (short timeout) should not emit heartbeats/start banners."""
    fake = _fake_popen(stdout_text="TensorRT version: 10.15.0\n")

    with (
        caplog.at_level("INFO", logger="modelopt.onnx"),
        patch("subprocess.Popen", return_value=fake),
    ):
        result = ort_utils._run_trtexec(timeout=5, heartbeat_interval=30.0)

    assert "TensorRT version" in result.stdout
    messages = [r.getMessage() for r in caplog.records]
    assert not any("Starting trtexec" in m for m in messages)
    assert not any("still running" in m for m in messages)
    assert not any("trtexec finished" in m for m in messages)


def test_run_trtexec_timeout_raises_with_captured_output():
    fake = _fake_popen(stdout_text="partial\n", hold_s=10.0)

    with (
        patch("subprocess.Popen", return_value=fake),
        pytest.raises(subprocess.TimeoutExpired) as exc_info,
    ):
        ort_utils._run_trtexec(timeout=0.05, heartbeat_interval=30.0)

    assert "partial" in (exc_info.value.output or "")


def test_run_trtexec_missing_binary_raises_file_not_found():
    with (
        patch("subprocess.Popen", side_effect=FileNotFoundError("missing")),
        pytest.raises(FileNotFoundError, match="trtexec"),
    ):
        ort_utils._run_trtexec(["--help"])
