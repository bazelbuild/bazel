# Copyright 2023 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SBOM generator.

This tool takes input from several sources and weaves together an SBOM.

Inputs:
  - the output of packages_used. This is a JSON block of license, package_info
    and other declarations, plus a list of all remote packages referenced.
  - TODO: the maven lock file
  - FUTURE: other packgage lock files
  - FUTURE: a user provided override of package URL to corrected information

This tool is private to the sbom() rule.
"""

import argparse
import datetime
import json


def create_sbom(package_info: dict) -> dict:  # pylint: disable=g-bare-generic
  """Creates a dict representing an SBOM.

  Args:
    package_info: dict of data from packages_used output
  Returns:
    dict of SBOM data
  """
  now = datetime.datetime.now(datetime.timezone.utc)
  ret = {
      "spdxVersion": "SPDX-2.3",
      "dataLicense": "CC0-1.0",
      "SPDXID": "SPDXRef-DOCUMENT",
      "documentNamespace": (
          "https://spdx.google/be852459-4c54-4c50-9d2f-0e48890418fc"
      ),
      "name": package_info["top_level_target"],
      "creationInfo": {
          "licenseListVersion": "",
          "creators": [
              "Tool: github.com/bazelbuild/bazel/tools/compliance/write_sbom",
              "Organization: Google LLC",
          ],
          "created": now.isoformat(),
      },
  }

  packages = []
  relationships = []

  relationships.append({
      "spdxElementId": "SPDXRef-DOCUMENT",
      "relatedSpdxElement": "SPDXRef-Package-main",
      "relationshipType": "DESCRIBES"
  })

  for pkg in package_info["packages"]:
    packages.append(
        {
            "name": pkg,
            # TODO(aiuto): Fill in the rest
            # "SPDXID": "SPDXRef-GooglePackage-4c7dc29872b9c418",
            # "supplier": "Organization: Google LLC",
            # "downloadLocation": "NOASSERTION",
            # "licenseConcluded": "License-da09db95a268defe",
            # "copyrightText": ""
        }
    )
    relationships.append(
        {
            "spdxElementId": "SPDXRef-Package-main",
            # "relatedSpdxElement": "SPDXRef-GooglePackage-4c7dc29872b9c418",
            "relationshipType": "CONTAINS",
        }
    )

  ret["packages"] = packages
  ret["relationships"] = relationships
  return ret


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Helper for creating SBOMs", fromfile_prefix_chars="@"
  )
  parser.add_argument(
      "--out", required=True, help="The output file, mandatory."
  )
  parser.add_argument(
      "--packages_used",
      required=True,
      help="JSON list of transitive package data for a target",
  )
  opts = parser.parse_args()

  with open(opts.packages_used, "rt", encoding="utf-8") as inp:
    package_info = json.loads(inp.read())
  with open(opts.out, "w", encoding="utf-8") as out:
    out.write(json.dumps(create_sbom(package_info), indent=2))


if __name__ == "__main__":
  main()
