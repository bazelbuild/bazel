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

"""Tests for pe_manifest."""

import os
import struct
import tempfile
import unittest

from pe_tools import KnownResourceTypes
from pe_tools import parse_pe
from src.tools.pe_manifest import pe_manifest


_FILE_ALIGNMENT = 0x200
_SECTION_ALIGNMENT = 0x1000
_TRAILER = b"trailing data"


def _minimal_executable(*, is_64_bit):
  optional_header_size = 0xF0 if is_64_bit else 0xE0
  optional_header_magic = 0x20B if is_64_bit else 0x10B
  number_of_directories_offset = 108 if is_64_bit else 92
  directories_offset = 112 if is_64_bit else 96
  machine = 0x8664 if is_64_bit else 0x14C

  pe_offset = 0x80
  coff_header_offset = pe_offset + 4
  optional_header_offset = coff_header_offset + 20
  section_header_offset = optional_header_offset + optional_header_size
  result = bytearray(_FILE_ALIGNMENT)
  result[:2] = b"MZ"
  struct.pack_into("<I", result, 0x3C, pe_offset)
  result[pe_offset : pe_offset + 4] = b"PE\0\0"
  struct.pack_into(
      "<HHIIIHH",
      result,
      coff_header_offset,
      machine,
      1,
      0,
      0,
      0,
      optional_header_size,
      0x22,
  )
  struct.pack_into("<H", result, optional_header_offset, optional_header_magic)
  struct.pack_into("<I", result, optional_header_offset + 4, _FILE_ALIGNMENT)
  struct.pack_into(
      "<II",
      result,
      optional_header_offset + 32,
      _SECTION_ALIGNMENT,
      _FILE_ALIGNMENT,
  )
  struct.pack_into("<I", result, optional_header_offset + 56, 0x2000)
  struct.pack_into("<I", result, optional_header_offset + 60, _FILE_ALIGNMENT)
  struct.pack_into("<H", result, optional_header_offset + 68, 3)
  struct.pack_into(
      "<I", result, optional_header_offset + number_of_directories_offset, 16
  )
  assert directories_offset + 16 * 8 == optional_header_size
  struct.pack_into(
      "<8sIIIIIIHHI",
      result,
      section_header_offset,
      b".text\0\0\0",
      1,
      0x1000,
      _FILE_ALIGNMENT,
      _FILE_ALIGNMENT,
      0,
      0,
      0,
      0,
      0x60000020,
  )
  result.extend(b"\xC3" + bytes(_FILE_ALIGNMENT - 1))
  result.extend(_TRAILER)
  return bytes(result)


class PeManifestTest(unittest.TestCase):

  def test_write_adds_resource_section(self):
    for is_64_bit in (False, True):
      with self.subTest(
          is_64_bit=is_64_bit
      ), tempfile.TemporaryDirectory() as td:
        executable = os.path.join(td, "test.exe")
        with open(executable, "wb") as output:
          output.write(_minimal_executable(is_64_bit=is_64_bit))

        first_manifest = b"<assembly>first</assembly>"
        pe_manifest.write_manifest(executable, first_manifest)

        with open(executable, "rb") as input_file:
          output = input_file.read()
        pe = parse_pe(output, verify_checksum=True)
        self.assertEqual(pe.file_header.NumberOfSections, 2)
        self.assertTrue(pe.checksum_correct)
        self.assertEqual(
            bytes(pe.parse_resources()[KnownResourceTypes.RT_MANIFEST][1][0]),
            first_manifest,
        )
        self.assertTrue(output.endswith(_TRAILER))

        second_manifest = b"<assembly>second</assembly>"
        pe_manifest.write_manifest(executable, second_manifest)

        with open(executable, "rb") as input_file:
          output = input_file.read()
        pe = parse_pe(output, verify_checksum=True)
        self.assertEqual(pe.file_header.NumberOfSections, 2)
        self.assertTrue(pe.checksum_correct)
        self.assertEqual(
            bytes(pe.parse_resources()[KnownResourceTypes.RT_MANIFEST][1][0]),
            second_manifest,
        )
        self.assertTrue(output.endswith(_TRAILER))


if __name__ == "__main__":
  unittest.main()
