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
"""Utility methods for turning package metadata to JSON.

These should eventually be part of rules_license.
"""

def _strip_null_repo(label):
    """Removes the null repo name (e.g. @//) from a string.

    The is to make str(label) compatible between bazel 5.x and 6.x
    """
    s = str(label)
    if s.startswith("@//"):
        return s[1:]
    elif s.startswith("@@//"):
        return s[2:]
    return s

def _bazel_package(label):
    """Returns the package containing a label."""
    clean_label = _strip_null_repo(label)
    return clean_label[0:-(len(label.name) + 1)]

_license_template = """{{
  "label": "{label}",
  "bazel_package": "{bazel_package}",
  "license_kinds": [{kinds}],
  "copyright_notice": "{copyright_notice}",
  "package_name": "{package_name}",
  "package_url": "{package_url}",
  "package_version": "{package_version}",
  "license_text": "{license_text}"
}}"""

_kind_template = """{{
  "target": "{kind_path}",
  "name": "{kind_name}",
  "conditions": {kind_conditions}
}}"""

def license_info_to_json(license):
    """Converts a LicenseInfo to JSON.

    Args:
        license: a LicenseInfo
    Returns:
        JSON representation of license.
    """
    kinds = []
    for kind in sorted(license.license_kinds, key = lambda x: x.name):
        kinds.append(_kind_template.format(
            kind_name = kind.name,
            kind_path = kind.label,
            kind_conditions = kind.conditions,
        ))

    return _license_template.format(
        copyright_notice = license.copyright_notice,
        kinds = ",".join(kinds),
        license_text = license.license_text.path,
        package_name = license.package_name,
        package_url = license.package_url,
        package_version = license.package_version,
        label = _strip_null_repo(license.label),
        bazel_package = _bazel_package(license.label),
    )

def licenses_to_json(licenses):
    """Converts a list of LicenseInfo to JSON.

    This list is sorted by label for stability.

    Args:
        licenses: list(LicenseInfo)
    Returns:
        JSON representation of licenses
    """
    all_licenses = []
    for license in sorted(licenses.to_list(), key = lambda x: x.label):
        all_licenses.append(license_info_to_json(license))
    return "[" + ",".join(all_licenses) + "]"

_package_info_template = """{{
  "target": "{label}",
  "bazel_package": "{bazel_package}",
  "package_name": "{package_name}",
  "package_url": "{package_url}",
  "package_version": "{package_version}"
}}"""

def package_info_to_json(package_info):
    """Converts a PackageInfo to json.

    Args:
        package_info: a PackageInfo
    Returns:
        JSON representation of package_info.
    """
    return _package_info_template.format(
        label = _strip_null_repo(package_info.label),
        bazel_package = _bazel_package(package_info.label),
        package_name = package_info.package_name,
        package_url = package_info.package_url,
        package_version = package_info.package_version,
    )

def package_infos_to_json(packages):
    """Converts a list of PackageInfo to JSON.

    This list is sorted by label for stability.

    Args:
        packages: list(PackageInfo)
    Returns:
        JSON representation of packages.
    """
    all_packages = []
    for package in sorted(packages.to_list(), key = lambda x: x.label):
        all_packages.append(package_info_to_json(package))
    return "[" + ",".join(all_packages) + "]"

def labels_to_json(labels):
    """Converts a list of Labels to JSON.

    This list is sorted for stability.

    Args:
        labels: list(Label)
    Returns:
        JSON representation of the labels.
    """
    return "[%s]" % ",".join(['"%s"' % _strip_null_repo(label) for label in sorted(labels)])
