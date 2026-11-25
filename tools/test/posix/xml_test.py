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

"""Tests the XML written by posix_tw --generate-xml."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


class PosixXmlTest(unittest.TestCase):

  def test_missing_log_and_xml_characters(self):
    with tempfile.TemporaryDirectory() as directory:
      log = Path(directory) / "test.log"
      xml = Path(directory) / "test.xml"
      env = dict(os.environ, TEST_BINARY="./a&\"<>\uffff", TEST_TOTAL_SHARDS="0")
      command = [WRAPPER, "--generate-xml", str(log), str(xml), "1", "0"]
      subprocess.run(command, env=env, check=True)
      suite = ET.parse(xml).getroot().find("testsuite")
      self.assertEqual(suite.get("name"), 'a&"<>?')
      self.assertEqual(suite.find("testcase").get("name"), 'a&"<>?')
      heading = (
          "\nGenerated test.log (if the file is not UTF-8, then this may be"
          " unreadable):\n"
      )
      self.assertEqual(suite.findtext("system-out"), heading + "\n      ")

      log.write_bytes(b"\n]]>\xc3\xa9\xef\xbf\xbe\x01\n\n")
      subprocess.run(command, env=env, check=True)
      output = ET.parse(xml).getroot().findtext("testsuite/system-out")
      self.assertEqual(output, heading + "\n]]>é??\n      ")


if __name__ == "__main__":
  WRAPPER = sys.argv.pop(1)
  unittest.main()
