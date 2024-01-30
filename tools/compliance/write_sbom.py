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
  - the maven lock file (maven_install.json)
  - FUTURE: other packgage lock files
  - FUTURE: a user provided override of package URL to corrected information

This tool is private to the sbom() rule.
"""

import argparse
import datetime
import hashlib
import json


# pylint: disable=g-bare-generic
def create_sbom(package_info: dict, maven_packages: dict) -> dict:
  """Creates a dict representing an SBOM.

  Args:
    package_info: dict of data from packages_used output.
    maven_packages: packages gleaned from Maven lock file.

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

  # This is bazel private shenanigans.
  magic_file_suffix = "//file:file"

  for pkg in package_info["packages"]:
    tmp_id = hashlib.md5()
    tmp_id.update(pkg.encode("utf-8"))
    spdxid = "SPDXRef-GooglePackage-%s" % tmp_id.hexdigest()
    pi = {
        "name": pkg,
        "downloadLocation": "NOASSERTION",
        "SPDXID": spdxid,
        # TODO(aiuto): Fill in the rest
        # "supplier": "Organization: Google LLC",
        # "licenseConcluded": "License-XXXXXX",
        # "copyrightText": ""
    }

    have_maven = None
    if pkg.startswith("@maven//:"):
      have_maven = maven_packages.get(pkg[9:])
    elif pkg.endswith(magic_file_suffix):
      # Bazel hacks jvm_external to add //file:file as a target, then we depend
      # on that rather than the correct thing.
      # Example: @org_apache_tomcat_tomcat_annotations_api_8_0_5//file:file
      # Check for just the versioned root
      have_maven = maven_packages.get(pkg[1 : -len(magic_file_suffix)])

    if have_maven:
      pi["downloadLocation"] = have_maven["url"]
    else:
      # TODO(aiuto): Do something better for this case.
      print("MISSING ", pkg)

    packages.append(pi)
    relationships.append({
        "spdxElementId": "SPDXRef-Package-main",
        "relatedSpdxElement": spdxid,
        "relationshipType": "CONTAINS",
    })

  ret["packages"] = packages
  ret["relationships"] = relationships
  return ret


def maven_to_bazel(s):
  """Returns a string with maven separators mapped to what we use in Bazel.

  Essentially '.', '-', ':' => '_'.

  Args:
    s: a string

  Returns:
    a string
  """
  return s.replace(".", "_").replace("-", "_").replace(":", "_")


# pylint: disable=g-bare-generic
def maven_install_to_packages(maven_install: dict) -> dict:
  """Convert raw maven lock file into a dict keyed by bazel package names.

  Args:
    maven_install: raw maven lock file data

  Returns:
    dict keyed by names created by rules_jvm_external
  """

  # Map repo coordinate back to the download repository.
  # The input dict is of the form
  # "https//repo1.maven.org/": [ com.google.foo:some.package, ...]
  # But.... sometimes the artifact is
  #    com.google.foo:some.package.jar.arch
  # and then  that means the artifact table has an entry
  # in their shasums table keyed by arch.

  repo_to_url = {}
  for url, repos in maven_install["repositories"].items():
    for repo in repos:
      if repo in repo_to_url:
        print(
            "WARNING: Duplicate download path for <%s>. Using %s"
            % (repo, repo_to_url[repo])
        )
        continue
      repo_to_url[repo] = url

  ret = {}
  for name, info in maven_install["artifacts"].items():
    repo, artifact = name.split(":")
    version = info["version"]

    for arch in info["shasums"].keys():
      # build the download URL
      sub_version = version
      repo_name = name
      if arch != "jar":
        sub_version = version + "-" + arch
        repo_name = "%s:jar:%s" % (name, arch)

      url = (
          "{mirror}{repo}/{artifact}/{version}/{artifact}-{version}.jar".format(
              mirror=repo_to_url[repo_name],
              repo=repo.replace(".", "/"),
              artifact=artifact,
              version=version,
          )
      )
      tmp = info.copy()
      tmp["maven_name"] = name
      tmp["url"] = url
      bazel_name = maven_to_bazel(name) + "_" + maven_to_bazel(sub_version)
      ret[bazel_name] = tmp
      if arch == "jar":
        ret[bazel_name] = tmp
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
  parser.add_argument(
      "--maven_install",
      required=False,
      default="",
      help="Maven lock file",
  )
  opts = parser.parse_args()

  with open(opts.packages_used, "rt", encoding="utf-8") as inp:
    package_info = json.loads(inp.read())

  maven_packages = None
  if opts.maven_install:
    with open(opts.maven_install, "rt", encoding="utf-8") as inp:
      maven_install = json.loads(inp.read())
      maven_packages = maven_install_to_packages(maven_install)
      # Useful for debugging
      # print(json.dumps(maven_packages, indent=2))

  sbom = create_sbom(package_info, maven_packages)
  with open(opts.out, "w", encoding="utf-8") as out:
    out.write(json.dumps(sbom, indent=2))


if __name__ == "__main__":
  main()
