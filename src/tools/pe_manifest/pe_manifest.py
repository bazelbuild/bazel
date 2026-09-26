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
import struct
import sys
import tempfile

import grope
from pe_tools import IMAGE_DIRECTORY_ENTRY_RESOURCE
from pe_tools import KnownResourceTypes
from pe_tools import parse_pe
from pe_tools import pe_resources_prepack


_RT_MANIFEST = KnownResourceTypes.RT_MANIFEST
_CREATEPROCESS_MANIFEST_RESOURCE_ID = 1
_IMAGE_SCN_CNT_INITIALIZED_DATA = 0x00000040
_IMAGE_SCN_MEM_READ = 0x40000000

_PE_OFFSET_OFFSET = 0x3C
_COFF_HEADER_SIZE = 20
_SECTION_HEADER_SIZE = 40


def _align(value, alignment):
  return (value + alignment - 1) // alignment * alignment


def _add_resource_section(executable, packed_resources):
  """Returns executable with a new resource section containing resources.

  References:
  https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#section-table-section-headers
  https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#optional-header-data-directories-image-only
  """
  if len(executable) < _PE_OFFSET_OFFSET + 4:
    raise ValueError("invalid PE header")

  pe_offset = struct.unpack_from("<I", executable, _PE_OFFSET_OFFSET)[0]
  if executable[pe_offset : pe_offset + 4] != b"PE\0\0":
    raise ValueError("invalid PE signature")

  coff_header_offset = pe_offset + 4
  number_of_sections = struct.unpack_from(
      "<H", executable, coff_header_offset + 2
  )[0]
  optional_header_size = struct.unpack_from(
      "<H", executable, coff_header_offset + 16
  )[0]
  optional_header_offset = coff_header_offset + _COFF_HEADER_SIZE
  optional_header_end = optional_header_offset + optional_header_size
  if optional_header_end > len(executable):
    raise ValueError("truncated PE optional header")

  optional_header_magic = struct.unpack_from(
      "<H", executable, optional_header_offset
  )[0]
  if optional_header_magic == 0x10B:
    number_of_directories_offset = optional_header_offset + 92
    directories_offset = optional_header_offset + 96
  elif optional_header_magic == 0x20B:
    number_of_directories_offset = optional_header_offset + 108
    directories_offset = optional_header_offset + 112
  else:
    raise ValueError("unsupported PE optional header")

  number_of_directories = struct.unpack_from(
      "<I", executable, number_of_directories_offset
  )[0]
  resource_directory_offset = (
      directories_offset + 8 * IMAGE_DIRECTORY_ENTRY_RESOURCE
  )
  if (
      number_of_directories <= IMAGE_DIRECTORY_ENTRY_RESOURCE
      or resource_directory_offset + 8 > optional_header_end
  ):
    raise ValueError("PE optional header has no resource directory entry")
  if struct.unpack_from("<II", executable, resource_directory_offset) != (0, 0):
    raise ValueError("PE resource directory is not empty")

  section_headers_offset = optional_header_end
  section_headers_end = (
      section_headers_offset + number_of_sections * _SECTION_HEADER_SIZE
  )
  new_section_headers_end = section_headers_end + _SECTION_HEADER_SIZE
  size_of_headers = struct.unpack_from(
      "<I", executable, optional_header_offset + 60
  )[0]

  sections = []
  for index in range(number_of_sections):
    section_offset = section_headers_offset + index * _SECTION_HEADER_SIZE
    if section_offset + _SECTION_HEADER_SIZE > len(executable):
      raise ValueError("truncated PE section table")
    virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from(
        "<IIII", executable, section_offset + 8
    )
    sections.append((virtual_size, virtual_address, raw_size, raw_offset))

  raw_sections = [section for section in sections if section[2]]
  if not sections or not raw_sections:
    raise ValueError("PE file has no initialized sections")
  first_raw_offset = min(section[3] for section in raw_sections)
  if new_section_headers_end > min(size_of_headers, first_raw_offset):
    raise ValueError("PE headers have no room for another section")

  section_alignment, file_alignment = struct.unpack_from(
      "<II", executable, optional_header_offset + 32
  )
  if not section_alignment or not file_alignment:
    raise ValueError("invalid PE section alignment")

  raw_offset = max(section[3] + section[2] for section in raw_sections)
  if raw_offset > len(executable) or raw_offset % file_alignment:
    raise ValueError("invalid PE section layout")
  last_virtual_size, last_virtual_address, _, _ = sections[-1]
  virtual_address = _align(
      last_virtual_address + last_virtual_size, section_alignment
  )
  section_data = packed_resources.pack(virtual_address)
  raw_size = _align(len(section_data), file_alignment)

  result = bytearray(executable)
  struct.pack_into(
      "<8sIIIIIIHHI",
      result,
      section_headers_end,
      b".rsrc\0\0\0",
      len(section_data),
      virtual_address,
      raw_size,
      raw_offset,
      0,
      0,
      0,
      0,
      _IMAGE_SCN_CNT_INITIALIZED_DATA | _IMAGE_SCN_MEM_READ,
  )
  struct.pack_into("<H", result, coff_header_offset + 2, number_of_sections + 1)
  size_of_initialized_data = struct.unpack_from(
      "<I", result, optional_header_offset + 8
  )[0]
  struct.pack_into(
      "<I",
      result,
      optional_header_offset + 8,
      size_of_initialized_data + raw_size,
  )
  struct.pack_into(
      "<I",
      result,
      optional_header_offset + 56,
      _align(virtual_address + len(section_data), section_alignment),
  )
  struct.pack_into(
      "<II",
      result,
      resource_directory_offset,
      virtual_address,
      len(section_data),
  )
  return (
      bytes(result[:raw_offset])
      + section_data
      + bytes(raw_size - len(section_data))
      + bytes(result[raw_offset:])
  )


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
      input_data = input_file.read()
      pe = parse_pe(input_data)
      resources = pe.parse_resources()
      add_resource_section = resources is None
      if resources is None:
        resources = {}
      elif not pe.is_dir_safely_resizable(IMAGE_DIRECTORY_ENTRY_RESOURCE):
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
      if pe.has_signature():
        pe.remove_signature()

      packed_resources = pe_resources_prepack(resources)
      if add_resource_section:
        pe = parse_pe(
            _add_resource_section(bytes(pe.to_blob()), packed_resources)
        )
      else:
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
