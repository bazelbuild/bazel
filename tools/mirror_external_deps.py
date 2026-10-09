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
"""Finds and mirrors external repository URLs to mirror.bazel.build.

Uses `bazel mod show_repo --all_repos --output=streamed_jsonproto` to inspect
all transitive repository definitions, plus any lockfiles referenced by
`repo_cache_tar` rules (such as `//src/test/tools/bzlmod:MODULE.bazel.lock`).
"""

import argparse
from concurrent import futures
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

MIRROR_HOST = "mirror.bazel.build"
MIRROR_BASE = f"https://{MIRROR_HOST}"
DEFAULT_GCS_BUCKET = "bazel-mirror"
USER_AGENT = "Bazel-Mirror/1.0"
IGNORED_MIRROR_HOSTS = frozenset({MIRROR_HOST, "bcr.cloudflaremirrors.com"})
_TRANSIENT_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})
_DOWNLOADER_REWRITE_RE = re.compile(
    r"^\s*rewrite\s+\(?([a-zA-Z0-9._-]+)\)?/.*https://mirror\.bazel\.build/",
    re.MULTILINE,
)


@dataclasses.dataclass(frozen=True)
class RepoUrlRef:
  """Tracks a repository that references a candidate URL."""

  repo_name: str
  explicit_mirrors: tuple[str, ...] = ()


def urlopen_with_retry(
    req: urllib.request.Request | str,
    timeout: int = 10,
    retries: int = 3,
) -> Any:
  """Opens a URL with exponential backoff on transient HTTP/network errors."""
  for attempt in range(retries):
    try:
      return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
      if e.code not in _TRANSIENT_HTTP_CODES or attempt == retries - 1:
        raise
    except (urllib.error.URLError, TimeoutError, OSError):
      if attempt == retries - 1:
        raise
    time.sleep(1 << attempt)


def parse_downloader_cfg_domains(workspace: pathlib.Path) -> set[str]:
  """Extracts mirrored domains from bazel_downloader.cfg in the workspace."""
  cfg_path = workspace / "bazel_downloader.cfg"
  if not cfg_path.is_file():
    return set()
  text = cfg_path.read_text(encoding="utf-8")
  return set(_DOWNLOADER_REWRITE_RE.findall(text))


def run_show_repo(
    workspace: pathlib.Path, bazel_bin: str
) -> list[dict[str, Any]]:
  """Runs `bazel mod show_repo --all_repos` and parses JSON proto lines."""
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


def get_explicit_attrs(repo: dict[str, Any]) -> dict[str, dict[str, Any]]:
  """Returns explicitly specified attributes keyed by attribute name."""
  return {
      a["name"]: a
      for a in repo.get("attribute", [])
      if a.get("explicitlySpecified")
  }


def fetch_archive_url(source_url: str) -> tuple[str | None, str | None]:
  """Fetches a BCR source.json and returns (archive_url, error_message)."""
  req = urllib.request.Request(source_url, headers={"User-Agent": USER_AGENT})
  try:
    with urlopen_with_retry(req, timeout=10) as resp:
      data = json.load(resp)
  except (
      urllib.error.URLError,
      TimeoutError,
      OSError,
      json.JSONDecodeError,
  ) as e:
    return None, f"{source_url}: {e}"

  if not isinstance(data, dict):
    return None, f"{source_url}: expected JSON object"
  # Non-archive repos (e.g. git_repository) legitimately omit "url".
  url = data.get("url")
  return (url if isinstance(url, str) else None), None


def extract_lockfile_source_urls(
    workspace: pathlib.Path, lockfile_labels: set[str]
) -> tuple[set[str], list[str]]:
  """Fetches archive URLs from lockfiles referenced by repo_cache_tar rules."""
  source_json_urls: set[str] = set()
  errors: list[str] = []

  for label in sorted(lockfile_labels):
    rel = label.removeprefix("@@").removeprefix("@").removeprefix("//")
    rel_path = pathlib.Path(rel.replace(":", "/").lstrip("/"))
    lock_path = workspace / rel_path
    if not lock_path.is_file():
      msg = f"Lockfile not found: {lock_path}"
      print(f"WARNING: {msg}", file=sys.stderr)
      errors.append(msg)
      continue
    try:
      lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
      msg = f"Failed to read lockfile {lock_path}: {e}"
      print(f"WARNING: {msg}", file=sys.stderr)
      errors.append(msg)
      continue
    for url in lock_data.get("registryFileHashes", {}):
      if url.endswith("/source.json"):
        source_json_urls.add(url)

  archive_urls: set[str] = set()
  if source_json_urls:
    with futures.ThreadPoolExecutor(max_workers=16) as ex:
      for url, err in ex.map(fetch_archive_url, sorted(source_json_urls)):
        if err:
          print(
              f"WARNING: Failed to fetch source.json {err}",
              file=sys.stderr,
          )
          errors.append(err)
        elif url:
          archive_urls.add(url)

  return archive_urls, errors


def collect_candidates(
    workspace: pathlib.Path,
    bazel_bin: str,
    domains: set[str],
    include_explicit_mirror_repos: bool,
) -> tuple[dict[str, list[RepoUrlRef]], set[str], list[str]]:
  """Collects candidate URLs to mirror from `show_repo` and lockfiles."""
  repos = run_show_repo(workspace, bazel_bin)

  candidates: dict[str, list[RepoUrlRef]] = {}
  explicit_mirrors: set[str] = set()
  lockfile_labels: set[str] = set()

  for repo in repos:
    cname = repo.get("canonicalName", "")
    rule_name = repo.get("repoRuleName", "")
    attrs = get_explicit_attrs(repo)

    if rule_name == "repo_cache_tar" and "lockfiles" in attrs:
      lockfile_labels.update(attrs["lockfiles"].get("stringListValue", []))

    urls: list[str] = []
    if "url" in attrs and attrs["url"].get("stringValue"):
      urls.append(attrs["url"]["stringValue"])
    if "urls" in attrs and attrs["urls"].get("stringListValue"):
      urls.extend(attrs["urls"]["stringListValue"])

    if not urls:
      continue

    repo_explicit_mirrors = tuple(
        u for u in urls if urllib.parse.urlparse(u).netloc == MIRROR_HOST
    )
    explicit_mirrors.update(repo_explicit_mirrors)

    for u in urls:
      if "{}" in u:
        continue
      host = urllib.parse.urlparse(u).netloc
      if host in IGNORED_MIRROR_HOSTS:
        continue
      if host in domains or (
          include_explicit_mirror_repos and repo_explicit_mirrors
      ):
        ref = RepoUrlRef(
            repo_name=cname, explicit_mirrors=repo_explicit_mirrors
        )
        candidates.setdefault(u, []).append(ref)

  archive_urls, errors = extract_lockfile_source_urls(
      workspace, lockfile_labels
  )
  for u in archive_urls:
    if urllib.parse.urlparse(u).netloc in domains:
      ref = RepoUrlRef(repo_name="repo_cache_tar(lockfiles)")
      candidates.setdefault(u, []).append(ref)

  return candidates, explicit_mirrors, errors


def to_mirror_url(source_url: str) -> str:
  """Converts a source URL to its corresponding mirror.bazel.build URL."""
  parsed = urllib.parse.urlparse(source_url)
  return f"{MIRROR_BASE}/{parsed.netloc}{parsed.path}"


def to_gcs_url(source_url: str, bucket: str) -> str:
  """Converts a source URL to its corresponding gs:// mirror bucket URL."""
  parsed = urllib.parse.urlparse(source_url)
  return f"gs://{bucket}/{parsed.netloc}{parsed.path}"


def head_status(url: str) -> tuple[str, int | str]:
  """Performs an HTTP HEAD request and returns (url, status_code_or_error)."""
  req = urllib.request.Request(
      url, method="HEAD", headers={"User-Agent": USER_AGENT}
  )
  try:
    with urlopen_with_retry(req, timeout=10) as resp:
      return url, resp.status
  except urllib.error.HTTPError as e:
    return url, e.code
  except (urllib.error.URLError, TimeoutError, OSError) as e:
    return url, str(e)


def find_missing_urls(
    candidates: dict[str, list[RepoUrlRef]], explicit_mirrors: set[str]
) -> list[str]:
  """Checks mirror.bazel.build via HEAD requests and returns missing URLs."""
  urls_to_check = set(explicit_mirrors) | {to_mirror_url(u) for u in candidates}
  with futures.ThreadPoolExecutor(max_workers=32) as ex:
    status_map = dict(ex.map(head_status, sorted(urls_to_check)))

  for url, status in sorted(status_map.items()):
    if status not in (200, 404):
      print(
          f"WARNING: Unexpected response from {url}: {status}",
          file=sys.stderr,
      )

  missing_urls: list[str] = []
  for u in sorted(candidates):
    if status_map.get(to_mirror_url(u)) == 200:
      continue
    refs = candidates[u]
    # Skip if every repo referencing this URL already has a working explicit
    # mirror.bazel.build URL.
    if all(
        any(status_map.get(m) == 200 for m in ref.explicit_mirrors)
        for ref in refs
    ):
      continue
    missing_urls.append(u)

  return missing_urls


def upload_to_gcs(source_url: str, bucket: str) -> tuple[str, str, str]:
  """Downloads source_url and uploads it to gs://<bucket>/<host>/<path>."""
  gcs_url = to_gcs_url(source_url, bucket)
  stat_proc = subprocess.run(
      ["gcloud", "storage", "ls", gcs_url],
      capture_output=True,
      text=True,
      check=False,
  )
  if stat_proc.returncode == 0:
    print(f"SKIPPED: {gcs_url} (already exists in GCS)")
    return "SKIPPED", gcs_url, "Artifact already exists"

  temp_filename = None
  try:
    req = urllib.request.Request(source_url, headers={"User-Agent": USER_AGENT})
    with (
        urlopen_with_retry(req, timeout=300) as resp,
        tempfile.NamedTemporaryFile(delete=False) as temp_file,
    ):
      temp_filename = temp_file.name
      hasher = hashlib.sha256()
      while chunk := resp.read(65536):
        temp_file.write(chunk)
        hasher.update(chunk)
      digest = hasher.hexdigest()

    subprocess.run(
        [
            "gcloud",
            "storage",
            "cp",
            "--no-clobber",
            "--cache-control=public, max-age=31536000",
            temp_filename,
            gcs_url,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    print(f"SUCCESS: {source_url} -> {gcs_url} (sha256: {digest})")
    return "SUCCESS", gcs_url, ""
  except (
      urllib.error.URLError,
      TimeoutError,
      OSError,
      subprocess.CalledProcessError,
  ) as e:
    if isinstance(e, subprocess.CalledProcessError) and e.stderr:
      detail = e.stderr.strip()
    else:
      detail = str(e)
    print(f"FAILED: {source_url} -> {gcs_url}: {detail}", file=sys.stderr)
    return "FAILED", source_url, detail
  finally:
    if temp_filename and os.path.exists(temp_filename):
      os.remove(temp_filename)


def print_urls(
    urls: list[str],
    candidates: dict[str, list[RepoUrlRef]],
    verbose: bool,
) -> None:
  """Prints URLs to stdout, optionally annotated with repository names."""
  for u in urls:
    if verbose:
      repo_names = sorted({ref.repo_name for ref in candidates[u]})
      print(f"{u}\t# {', '.join(repo_names)}")
    else:
      print(u)


def main() -> None:
  """Main entry point."""
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
      default=None,
      help=(
          "Domains to mirror (default: parsed from bazel_downloader.cfg, plus"
          " any repo explicitly listing mirror.bazel.build in its urls)."
      ),
  )
  parser.add_argument(
      "--include-mirrored",
      action="store_true",
      help=(
          "Print all mirrorable URLs without filtering out already-mirrored"
          " ones."
      ),
  )
  parser.add_argument(
      "--upload",
      action="store_true",
      help=(
          "Upload missing URLs to the GCS mirror bucket (requires gcloud"
          " storage)."
      ),
  )
  parser.add_argument(
      "--gcs-bucket",
      default=DEFAULT_GCS_BUCKET,
      help="GCS bucket name when --upload is set (default: %(default)s).",
  )
  parser.add_argument(
      "--verbose",
      "-v",
      action="store_true",
      help="Print canonical repository names alongside each URL.",
  )
  args = parser.parse_args()

  if args.domains is not None:
    domains = set(args.domains)
    include_explicit_mirror_repos = False
  else:
    domains = parse_downloader_cfg_domains(args.workspace)
    include_explicit_mirror_repos = True

  candidates, explicit_mirrors, lockfile_errors = collect_candidates(
      args.workspace,
      args.bazel,
      domains,
      include_explicit_mirror_repos,
  )

  if args.include_mirrored:
    print_urls(sorted(candidates), candidates, args.verbose)
    if lockfile_errors:
      sys.exit(1)
    return

  missing_urls = find_missing_urls(candidates, explicit_mirrors)

  if not args.upload:
    print_urls(missing_urls, candidates, args.verbose)
    if lockfile_errors:
      sys.exit(1)
    return

  if not missing_urls:
    print("All URLs are already mirrored. Nothing to upload.")
    if lockfile_errors:
      sys.exit(1)
    return

  print(
      f"Uploading {len(missing_urls)} missing URL(s) to"
      f" gs://{args.gcs_bucket} ..."
  )
  results = [upload_to_gcs(u, args.gcs_bucket) for u in missing_urls]
  successes = [r for r in results if r[0] == "SUCCESS"]
  skips = [r for r in results if r[0] == "SKIPPED"]
  failures = [r for r in results if r[0] == "FAILED"]
  print(
      f"\nMirroring complete. Success: {len(successes)},"
      f" Skipped: {len(skips)}, Failed: {len(failures)}"
  )
  if failures or lockfile_errors:
    sys.exit(1)


if __name__ == "__main__":
  main()
