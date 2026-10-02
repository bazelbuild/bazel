#!/usr/bin/env python3
#
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
"""Finds external repository URLs that should be mirrored to mirror.bazel.build.

Uses `bazel mod show_repo --all_repos --output=streamed_jsonproto` to inspect
all transitive repository definitions, plus any lockfiles referenced by
`repo_cache_tar` rules (such as `//src/test/tools/bzlmod:MODULE.bazel.lock`).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import pathlib
import subprocess
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

MIRROR_BASE = "https://mirror.bazel.build"
DEFAULT_DOMAINS = ("github.com", "maven.google.com", "cdn.azul.com")


def run_show_repo(workspace: pathlib.Path, bazel_bin: str) -> list[dict]:
  cmd = [
      bazel_bin,
      "mod",
      "show_repo",
      "--all_repos",
      "--output=streamed_jsonproto",
  ]
  proc = subprocess.run(
      cmd,
      cwd=workspace,
      capture_output=True,
      text=True,
      check=True,
  )
  return [
      json.loads(line)
      for line in proc.stdout.splitlines()
      if line.strip().startswith("{")
  ]


def get_attr_map(repo: dict) -> dict[str, dict]:
  return {
      a["name"]: a
      for a in repo.get("attribute", [])
      if a.get("explicitlySpecified")
  }


def extract_lockfile_source_urls(
    workspace: pathlib.Path, lockfile_labels: set[str]
) -> set[str]:
  """Fetches source.json files from lockfiles referenced by repo_cache_tar."""
  source_json_urls = set()
  for label in lockfile_labels:
    rel = label.removeprefix("@@").removeprefix("@").removeprefix("//")
    rel_path = pathlib.Path(rel.replace(":", "/").lstrip("/"))
    lock_path = workspace / rel_path
    if not lock_path.is_file():
      continue
    lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
    for url in lock_data.get("registryFileHashes", {}):
      if url.endswith("/source.json"):
        source_json_urls.add(url)

  def fetch_archive_url(source_url: str) -> str | None:
    try:
      with urlopen(source_url, timeout=10) as resp:
        data = json.load(resp)
        return data.get("url")
    except Exception:
      return None

  archive_urls = set()
  if source_json_urls:
    with ThreadPoolExecutor(max_workers=16) as ex:
      for u in ex.map(fetch_archive_url, sorted(source_json_urls)):
        if u:
          archive_urls.add(u)
  return archive_urls


def head_status(url: str) -> tuple[str, int | str]:
  req = Request(url, method="HEAD")
  try:
    with urlopen(req, timeout=10) as resp:
      return url, resp.status
  except HTTPError as e:
    return url, e.code
  except URLError as e:
    return url, str(e.reason)
  except Exception as e:
    return url, str(e)


def to_mirror_url(source_url: str) -> str:
  parsed = urlparse(source_url)
  return f"{MIRROR_BASE}/{parsed.netloc}{parsed.path}"


def main():
  default_workspace = pathlib.Path(
      os.environ.get("BUILD_WORKSPACE_DIRECTORY", pathlib.Path.cwd())
  )
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--workspace",
      type=pathlib.Path,
      default=default_workspace,
      help="Path to the Bazel workspace root (default: %(default)s).",
  )
  parser.add_argument(
      "--bazel",
      default="bazel",
      help="Bazel binary to invoke (default: bazel).",
  )
  parser.add_argument(
      "--domains",
      nargs="+",
      default=list(DEFAULT_DOMAINS),
      help=(
          "Domains to mirror (default: %(default)s, matching"
          " bazel_downloader.cfg)."
      ),
  )
  parser.add_argument(
      "--all-python-versions",
      action="store_true",
      help=(
          "Include all Python versions registered by transitive modules in"
          " rules_python, not just the workspace's default_python_version."
      ),
  )
  parser.add_argument(
      "--include-mirrored",
      action="store_true",
      help="Print all mirrorable URLs without filtering out already-mirrored ones.",
  )
  parser.add_argument(
      "--verbose",
      "-v",
      action="store_true",
      help="Print canonical repository names alongside each URL.",
  )
  args = parser.parse_args()

  domains = set(args.domains)
  repos = run_show_repo(args.workspace, args.bazel)

  default_py_prefix = None
  if not args.all_python_versions:
    for r in repos:
      if r.get("canonicalName") == "rules_python++python+pythons_hub":
        attrs = get_attr_map(r)
        ver = attrs.get("default_python_version", {}).get("stringValue")
        if ver:
          default_py_prefix = f"rules_python++python+python_{ver.replace('.', '_')}_"
        break

  candidates: dict[str, list[tuple[str, list[str]]]] = {}
  explicit_mirrors: set[str] = set()
  lockfile_labels: set[str] = set()

  for r in repos:
    cname = r.get("canonicalName", "")
    rule_name = r.get("repoRuleName", "")
    attrs = get_attr_map(r)

    if rule_name == "repo_cache_tar" and "lockfiles" in attrs:
      lockfile_labels.update(attrs["lockfiles"].get("stringListValue", []))

    if (
        default_py_prefix
        and cname.startswith("rules_python++python+python_")
        and not cname.startswith(default_py_prefix)
    ):
      continue

    urls = []
    if "url" in attrs and attrs["url"].get("stringValue"):
      urls.append(attrs["url"]["stringValue"])
    if "urls" in attrs and attrs["urls"].get("stringListValue"):
      urls.extend(attrs["urls"]["stringListValue"])

    if not urls:
      continue

    repo_explicit_mirrors = [
        u for u in urls if urlparse(u).netloc == "mirror.bazel.build"
    ]
    explicit_mirrors.update(repo_explicit_mirrors)

    for u in urls:
      if "{}" in u:
        continue
      if urlparse(u).netloc in domains:
        candidates.setdefault(u, []).append((cname, repo_explicit_mirrors))

  for u in extract_lockfile_source_urls(args.workspace, lockfile_labels):
    if urlparse(u).netloc in domains:
      candidates.setdefault(u, []).append(("repo_cache_tar(lockfiles)", []))

  if args.include_mirrored:
    for u in sorted(candidates):
      if args.verbose:
        repo_names = sorted({c for c, _ in candidates[u]})
        print(f"{u}\t# {', '.join(repo_names)}")
      else:
        print(u)
    return

  urls_to_check = set(explicit_mirrors) | {to_mirror_url(u) for u in candidates}
  with ThreadPoolExecutor(max_workers=32) as ex:
    status_map = dict(ex.map(head_status, sorted(urls_to_check)))

  for u in sorted(candidates):
    mirror_url = to_mirror_url(u)
    if status_map.get(mirror_url) == 200:
      continue
    repo_infos = candidates[u]
    if all(
        any(status_map.get(em) == 200 for em in ems) for _, ems in repo_infos
    ):
      continue
    if args.verbose:
      repo_names = sorted({c for c, _ in repo_infos})
      print(f"{u}\t# {', '.join(repo_names)}")
    else:
      print(u)


if __name__ == "__main__":
  main()
