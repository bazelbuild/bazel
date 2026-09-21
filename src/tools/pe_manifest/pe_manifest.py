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

"""Reads or replaces the application manifest in a Windows executable."""

import argparse
import os
import stat
import sys
import tempfile

import grope
from pe_tools import IMAGE_DIRECTORY_ENTRY_RESOURCE
from pe_tools import KnownResourceTypes
from pe_tools import parse_pe
from pe_tools import pe_resources_prepack


_RT_MANIFEST = KnownResourceTypes.RT_MANIFEST
_CREATEPROCESS_MANIFEST_RESOURCE_ID = 1


def _manifest_entries(resources):
  """Returns the language-to-data map for the application manifest, if any."""
  return resources.get(_RT_MANIFEST, {}).get(
      _CREATEPROCESS_MANIFEST_RESOURCE_ID
  )


def read_manifest(executable):
  """Writes the executable's application manifest to stdout."""
  with open(executable, "rb") as input_file:
    resources = parse_pe(grope.wrap_io(input_file)).parse_resources()
    manifests = _manifest_entries(resources or {})
    if not manifests:
      raise ValueError(f"{executable} does not contain an application manifest")
    # Application manifests normally have a single language. Match the Windows
    # resource APIs and use the first entry if a binary contains more than one.
    sys.stdout.buffer.write(bytes(next(iter(manifests.values()))))
    sys.stdout.buffer.flush()


def write_manifest(executable, manifest):
  """Replaces the executable's application manifest."""
  mode = stat.S_IMODE(os.stat(executable).st_mode)
  output_directory = os.path.dirname(os.path.abspath(executable))
  output_fd, output_path = tempfile.mkstemp(dir=output_directory)
  try:
    with open(executable, "rb") as input_file, os.fdopen(
        output_fd, "w+b"
    ) as output_file:
      pe = parse_pe(grope.wrap_io(input_file))
      resources = pe.parse_resources()
      if resources is None:
        raise ValueError(f"{executable} does not have a resource section")
      if not pe.is_dir_safely_resizable(IMAGE_DIRECTORY_ENTRY_RESOURCE):
        raise ValueError(
            f"the resource section of {executable} cannot be safely resized"
        )

      manifests_by_id = resources.setdefault(_RT_MANIFEST, {})
      manifests = manifests_by_id.setdefault(
          _CREATEPROCESS_MANIFEST_RESOURCE_ID, {}
      )
      if manifests:
        for language in manifests:
          manifests[language] = manifest
      else:
        # Application manifests are language neutral if no language was
        # specified by the linker.
        manifests[0] = manifest

      # Modifying any byte covered by Authenticode invalidates the signature.
      # pe_tools preserves any other data in the PE overlay.
      if pe.has_signature():
        pe.remove_signature()

      packed_resources = pe_resources_prepack(resources)
      resource_address = pe.resize_directory(
          IMAGE_DIRECTORY_ENTRY_RESOURCE, packed_resources.size
      )
      pe.set_directory(
          IMAGE_DIRECTORY_ENTRY_RESOURCE,
          packed_resources.pack(resource_address),
      )
      grope.dump(pe.to_blob(update_checksum=True), output_file)

    os.chmod(output_path, mode)
    os.replace(output_path, executable)
  except BaseException:
    try:
      os.close(output_fd)
    except OSError:
      pass
    try:
      os.unlink(output_path)
    except OSError:
      pass
    raise


def main():
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="command", required=True)

  read_parser = subparsers.add_parser("read")
  read_parser.add_argument("executable")

  write_parser = subparsers.add_parser("write")
  write_parser.add_argument("executable")
  write_parser.add_argument("manifest", nargs="?")

  args = parser.parse_args()
  try:
    if args.command == "read":
      read_manifest(args.executable)
    else:
      if args.manifest:
        with open(args.manifest, "rb") as manifest_file:
          manifest = manifest_file.read()
      else:
        manifest = sys.stdin.buffer.read()
      write_manifest(args.executable, manifest)
  except (OSError, RuntimeError, ValueError) as error:
    parser.exit(1, f"pe_manifest: {error}\n")


if __name__ == "__main__":
  main()
