# Copyright 2026 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Merges reproducible extension facts into a generated distfile lockfile."""

from __future__ import annotations

from collections.abc import Sequence
import json
import sys

_FACT_KEYS = ("facts", "factsVersions")


def merge_lockfiles(
    generated_path: str,
    checked_in_path: str,
    output_path: str,
) -> None:
  """Carries over non-null extension facts from the checked-in lockfile.

  Args:
    generated_path: Path to the freshly resolved MODULE.bazel.lock.
    checked_in_path: Path to the checked-in MODULE.bazel.lock.
    output_path: Destination path for the merged MODULE.bazel.lock.dist.
  """
  with open(generated_path, encoding="utf-8") as generated_file:
    merged = json.load(generated_file)
  with open(checked_in_path, encoding="utf-8") as checked_in_file:
    checked_in = json.load(checked_in_file)

  for key in _FACT_KEYS:
    value = checked_in.get(key)
    if value is not None:
      merged[key] = value

  with open(output_path, "w", encoding="utf-8", newline="\n") as output_file:
    json.dump(merged, output_file, indent=2, ensure_ascii=False)
    output_file.write("\n")


def main(argv: Sequence[str]) -> int:
  if len(argv) != 4:
    sys.stderr.write(
        f"Usage: {argv[0]} <generated_lockfile> <checked_in_lockfile>"
        " <output_lockfile>\n"
    )
    return 2
  merge_lockfiles(argv[1], argv[2], argv[3])
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
