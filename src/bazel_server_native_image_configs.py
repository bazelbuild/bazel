#!/usr/bin/env python3
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
"""Generates native-image configuration derived from BazelServer_deploy.jar."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile


DYNAMIC_PROXY_CONFIG = [["java.lang.reflect.TypeVariable"]]

ALL_MEMBERS = {
    "allDeclaredConstructors": True,
    "allDeclaredFields": True,
    "allDeclaredMethods": True,
}

CONSTRUCTORS_ONLY = {
    "allDeclaredConstructors": True,
}

METHODS_ONLY = {
    "allDeclaredMethods": True,
    "allPublicMethods": True,
}

FIELDS_ONLY = {
    "allDeclaredFields": True,
}

STARLARK_MEMBERS = {
    "allDeclaredConstructors": True,
    "allDeclaredFields": True,
    "allDeclaredMethods": True,
    "allPublicMethods": True,
}


def class_name(entry: str) -> str:
  return remove_suffix(entry, ".class").replace("/", ".")


def simple_class_name(entry: str) -> str:
  return remove_suffix(entry.rsplit("/", 1)[-1], ".class")


def remove_suffix(value: str, suffix: str) -> str:
  if value.endswith(suffix):
    return value[: -len(suffix)]
  return value


def class_entry(entry: str) -> bool:
  return entry.endswith(".class")


def under_any(entry: str, prefixes: tuple[str, ...]) -> bool:
  return any(entry.startswith(prefix) for prefix in prefixes)


def emit_json(path: str, value: object) -> None:
  with Path(path).open("w", encoding="utf-8") as output:
    json.dump(value, output, indent=2, sort_keys=True)
    output.write("\n")


def reflection_entry(entry: str, attributes: dict[str, bool]) -> dict[str, object]:
  return {"name": class_name(entry), **attributes}


def emit_reflection_config(
    path: str,
    entries: list[str],
    predicate,
    attributes: dict[str, bool],
) -> None:
  emit_json(
      path,
      [
          reflection_entry(entry, attributes)
          for entry in entries
          if class_entry(entry) and predicate(entry)
      ],
  )


def emit_proto_reflection_config(path: str, entries: list[str]) -> None:
  proto_prefixes = {
      remove_suffix(entry, "OrBuilder.class")
      for entry in entries
      if entry.endswith("OrBuilder.class")
  }

  def is_proto_class(entry: str) -> bool:
    if not class_entry(entry):
      return False
    class_path = remove_suffix(entry, ".class")
    if class_path in proto_prefixes:
      return True
    if class_path.endswith("OrBuilder"):
      return remove_suffix(class_path, "OrBuilder") in proto_prefixes
    nested_class = class_path
    while "$" in nested_class:
      nested_class = nested_class.rsplit("$", 1)[0]
      if nested_class in proto_prefixes:
        return True
    return False

  emit_json(
      path,
      [reflection_entry(entry, ALL_MEMBERS) for entry in entries if is_proto_class(entry)],
  )


def main(argv: list[str]) -> int:
  if len(argv) != 15:
    print(
        "usage: bazel_server_native_image_configs "
        "<server_deploy.jar> <dynamic_proxy.json> <12 reflect config outputs>",
        file=sys.stderr,
    )
    return 2

  (
      server_deploy_jar,
      dynamic_proxy_config,
      caffeine_reflect_config,
      proto_reflect_config,
      options_reflect_config,
      converter_reflect_config,
      rule_reflect_config,
      starlark_reflect_config,
      bazel_methods_reflect_config,
      bazel_fields_reflect_config,
      gson_reflect_config,
      bzlmod_reflect_config,
      netty_jctools_reflect_config,
      netty_buffer_reflect_config,
  ) = argv[1:]

  with ZipFile(server_deploy_jar) as deploy_jar:
    entries = sorted({info.filename for info in deploy_jar.infolist() if not info.is_dir()})

  emit_json(dynamic_proxy_config, DYNAMIC_PROXY_CONFIG)
  emit_proto_reflection_config(proto_reflect_config, entries)
  emit_reflection_config(
      options_reflect_config,
      entries,
      lambda entry: under_any(
          entry,
          (
              "com/google/devtools/build/",
              "com/google/devtools/common/",
          ),
      )
      and re.search(r"(^|/)[^/]*(Options|Option)[^/]*[.]class$", entry),
      ALL_MEMBERS,
  )
  emit_reflection_config(
      converter_reflect_config,
      entries,
      lambda entry: re.search(r"(^|/)[^/]*Converter[^/]*[.]class$", entry),
      ALL_MEMBERS,
  )
  emit_reflection_config(
      rule_reflect_config,
      entries,
      lambda entry: under_any(
          entry,
          (
              "com/google/devtools/build/lib/analysis/",
              "com/google/devtools/build/lib/rules/",
              "com/google/devtools/build/lib/bazel/rules/",
          ),
      ),
      CONSTRUCTORS_ONLY,
  )
  emit_reflection_config(
      starlark_reflect_config,
      entries,
      lambda entry: under_any(
          entry,
          (
              "net/starlark/java/annot/",
              "net/starlark/java/eval/",
              "net/starlark/java/syntax/",
              "net/starlark/java/lib/",
          ),
      ),
      STARLARK_MEMBERS,
  )
  emit_reflection_config(
      bazel_methods_reflect_config,
      entries,
      lambda entry: under_any(
          entry,
          (
              "com/google/devtools/build/lib/",
              "com/google/devtools/common/options/",
          ),
      ),
      METHODS_ONLY,
  )
  emit_reflection_config(
      bazel_fields_reflect_config,
      entries,
      lambda entry: under_any(
          entry,
          (
              "com/google/devtools/build/lib/",
              "com/google/devtools/common/options/",
          ),
      ),
      FIELDS_ONLY,
  )
  emit_reflection_config(
      gson_reflect_config,
      entries,
      lambda entry: simple_class_name(entry).startswith("GsonTypeAdapter")
      or simple_class_name(entry).endswith("_GsonTypeAdapter")
      or simple_class_name(entry).endswith("$GsonTypeAdapter"),
      ALL_MEMBERS,
  )
  emit_reflection_config(
      bzlmod_reflect_config,
      entries,
      lambda entry: entry.startswith("com/google/devtools/build/lib/bazel/bzlmod/"),
      ALL_MEMBERS,
  )
  emit_reflection_config(
      caffeine_reflect_config,
      entries,
      lambda entry: re.fullmatch(
          r"com/github/benmanes/caffeine/cache/[A-Z0-9]+[.]class", entry
      ),
      ALL_MEMBERS,
  )
  emit_reflection_config(
      netty_jctools_reflect_config,
      entries,
      lambda entry: entry.startswith(
          "io/netty/util/internal/shaded/org/jctools/queues/"
      ),
      FIELDS_ONLY,
  )
  emit_reflection_config(
      netty_buffer_reflect_config,
      entries,
      lambda entry: entry.startswith("io/netty/buffer/"),
      ALL_MEMBERS,
  )
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv))
