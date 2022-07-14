# pylint: disable=g-direct-third-party-import
# Copyright 2022 The Bazel Authors. All rights reserved.
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
"""AndroidManifest tool to enforce a floor on the minSdkVersion attribute.

Ensures that the minSdkVersion attribute is >= than the specified floor,
and if the attribute is either not specified or less than the floor,
sets it to the floor.
"""

import os
import sys

# Do not edit this line. Copybara replaces it with PY2 migration helper.
from absl import app
from absl import flags
from lxml import etree

BUMP = "bump"
VALIDATE = "validate"

USES_SDK = "uses-sdk"
MIN_SDK_ATTRIB = "{http://schemas.android.com/apk/res/android}minSdkVersion"

_ACTION = flags.DEFINE_enum(
    "action",
    None,
    [BUMP, VALIDATE],
    f"Action to perform, either {BUMP} or {VALIDATE}")
_MANIFEST = flags.DEFINE_string(
    "manifest",
    None,
    "AndroidManifest.xml of the instrumentation APK")
_MIN_SDK_FLOOR = flags.DEFINE_integer(
    "min_sdk_floor",
    0,
    "Min SDK floor",
    lower_bound=0)
_OUTPUT = flags.DEFINE_string(
    "output",
    None,
    f"Output AndroidManifest.xml to generate, only needed for {BUMP}")
_LOG = flags.DEFINE_string("log", None, "Path to write the log to")


class MinSdkError(Exception):
  """Raised when there is a problem with the min SDK attribute in AndroidManifest.xml."""


def _BumpMinSdk(xml_content, min_sdk_floor):
  """Checks the min SDK in xml_content and replaces with min_sdk_floor if needed.

  Args:
    xml_content: str, the contents of the AndroidManifest.xml file
    min_sdk_floor: int, the min SDK floor

  Returns:
    A tuple with the following elements:
    - str: The xml contents of the manifest with the min SDK floor enforced.
      This string will be equal to the input if the min SDK is already not less
      than the floor.
    - str: log message of action taken
  """
  if min_sdk_floor == 0:
    return xml_content, "No min SDK floor specified. Manifest unchanged."

  root = etree.fromstring(xml_content)

  uses_sdk = root.find(USES_SDK)
  if uses_sdk is None:
    etree.SubElement(root, USES_SDK, {MIN_SDK_ATTRIB: str(min_sdk_floor)})
    return (
        etree.tostring(root, xml_declaration=True, encoding="utf-8"),
        "No uses-sdk element found while floor is specified "
        + f"({min_sdk_floor}). Min SDK added.")

  min_sdk = uses_sdk.get(MIN_SDK_ATTRIB)
  if min_sdk is None:
    uses_sdk.set(MIN_SDK_ATTRIB, str(min_sdk_floor))
    return (
        etree.tostring(root, xml_declaration=True, encoding="utf-8"),
        "No minSdkVersion attribute found while floor is specified"
        + f"({min_sdk_floor}). Min SDK added.")

  try:
    min_sdk_int = int(min_sdk)
  except ValueError:
    return (
        xml_content,
        f"Placeholder used for the minSdkVersion attribute ({min_sdk}). "
        + "Manifest unchanged.")

  if min_sdk_int < min_sdk_floor:
    uses_sdk.set(MIN_SDK_ATTRIB, str(min_sdk_floor))
    return (
        etree.tostring(root, xml_declaration=True, encoding="utf-8"),
        f"minSdkVersion attribute specified in the manifest ({min_sdk}) "
        + f"is less than the floor ({min_sdk_floor}). Min SDK replaced.")
  return (
      xml_content,
      f"minSdkVersion attribute specified in the manifest ({min_sdk}) "
      + f"is not less than the floor ({min_sdk_floor}). Manifest unchanged.")


def _ValidateMinSdk(xml_content, min_sdk_floor):
  """Checks the min SDK in xml_content and raises MinSdkError if it is either not specified or less than the floor.

  Args:
    xml_content: str, the contents of the AndroidManifest.xml file
    min_sdk_floor: int, the min SDK floor
  Returns:
    str: log message
  Raises:
    MinSdkError: The min SDK is less than the specified floor.
  """
  if min_sdk_floor == 0:
    return "No min SDK floor specified."

  root = etree.fromstring(xml_content)

  uses_sdk = root.find(USES_SDK)
  if uses_sdk is None:
    raise MinSdkError(
        "No uses-sdk element found in manifest "
        + f"while floor is specified ({min_sdk_floor}).")

  min_sdk = uses_sdk.get(MIN_SDK_ATTRIB)
  if min_sdk is None:
    raise MinSdkError(
        "No minSdkVersion attribute found in manifest "
        + f"while floor is specified ({min_sdk_floor}).")

  try:
    min_sdk_int = int(min_sdk)
  except ValueError:
    return f"Placeholder minSdkVersion = {min_sdk}\n min SDK floor = {min_sdk_floor}"

  if min_sdk_int < min_sdk_floor:
    raise MinSdkError(
        f"minSdkVersion attribute specified in  the manifest ({min_sdk}) "
        + f"is less than the floor ({min_sdk_floor}).")
  return f"minSdkVersion = {min_sdk}\n min SDK floor = {min_sdk_floor}"


def main(unused_argv):
  manifest_path = _MANIFEST.value
  with open(manifest_path, "rb") as f:
    manifest = f.read()

  if _ACTION.value == BUMP:
    output_path = _OUTPUT.value
    dirname = os.path.dirname(output_path)
    if not os.path.exists(dirname):
      os.makedirs(dirname)

    out_contents, log_message = _BumpMinSdk(manifest, _MIN_SDK_FLOOR.value)
    with open(output_path, "wb") as f:
      f.write(out_contents)

  elif _ACTION.value == VALIDATE:
    try:
      log_message = _ValidateMinSdk(manifest, _MIN_SDK_FLOOR.value)
    except MinSdkError as e:
      sys.exit(str(e))
  else:
    sys.exit(f"Action must be either {BUMP} or {VALIDATE}")

  if _LOG.value is not None:
    log_path = _LOG.value
    dirname = os.path.dirname(log_path)
    if not os.path.exists(dirname):
      os.makedirs(dirname)
    with open(log_path, "w") as f:
      f.write(log_message)

if __name__ == "__main__":
  app.run(main)
