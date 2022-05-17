# Copyright 2020 Google LLC
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

"""Providers for license rules."""

LicenseKindInfo = provider(
    doc = """Provides information about a license kind.""",
    fields = {
        "conditions": "List of conditions to be met when using this software.",
        "label": "The full path to the license kind definition.",
        "name": "License Name",
    },
)

LicenseInfo = provider(
    doc = """Provides information about an instance of a license.""",
    fields = {
        "copyright_notice": "Human readable short copyright notice",
        "license_kinds": "License kinds",
        "license_text": "License file",
        "package_name": "Human readable package name",
        "rule": "From whence this came",
    },
)

LicensesInfo = provider(
    doc = """The set of license instances used in a target.""",
    fields = {
        "licenses": "list(LicenseInfo).",
    },
)
